"""Classification experiments on frozen CLIP features (output of extract_features.py).

Every step below is optional and they chain in this order:

  1. zero-shot head        text embeddings of the class prompts (always the starting point)
  2. --wise-alpha A        linear probe trained on CIFAR-10 train features, initialised at the
                           zero-shot head, then interpolated with it (WiSE-FT style, head only).
                           A=0 keeps the pure zero-shot head, A=1 keeps the pure linear probe.
  3. --pseudo-rounds R     transductive self-training on the unlabeled test features: keep the
                           most confident test predictions of each class as pseudo-labels,
                           fit a head on them (anchored to the current head), repeat R times.
  4. --knn K               smooth test probabilities with their K nearest test neighbours
                           (--knn-iters > 1: iterative label propagation on that graph).
  5. --balance             rebalance test predictions towards a uniform class distribution
                           (Sinkhorn). Assumes a balanced test set: state it in the report.

No test label is ever used. Validation accuracy is reported for the head (steps 1-3) and
measures in-domain accuracy only, not test-domain accuracy.

With several --features directories, every step runs on the probability average of the
models: the ensemble picks the pseudo-labels and each model refits its own head on them,
and kNN uses the concatenation of the models' normalized features.

Example:
    python feature_experiments.py --features features/bigg_laion2b --output runs/feat_zs
    python feature_experiments.py --features features/bigg_laion2b features/vith_dfn5b \
        --output runs/ens_zs
    python feature_experiments.py --features features/bigg_laion2b --output runs/feat_pl \
        --pseudo-rounds 3 --balance
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

CLASS_NAMES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
SCALE = 100.0  # CLIP logit scale


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--features", type=Path, nargs="+", required=True,
        help="One or more directories written by extract_features.py. Several = probability ensemble.",
    )
    p.add_argument("--model-weights", type=float, nargs="+", default=None, help="Ensemble weights (default: equal)")
    p.add_argument("--data", type=Path, default=Path("data"), help="Only used for sample_submission.csv")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--head", default="openai_imagenet", help="Zero-shot head stored in heads.npz")
    # linear probe / WiSE
    p.add_argument("--wise-alpha", type=float, default=0.0)
    # pseudo-labelling
    p.add_argument("--pseudo-rounds", type=int, default=0)
    p.add_argument("--pseudo-frac", type=float, default=0.5, help="Fraction of each predicted class kept")
    p.add_argument("--pseudo-with-train", action="store_true", help="Also fit on labeled CIFAR train features")
    p.add_argument(
        "--pseudo-balance", action="store_true",
        help="Rebalance test probabilities (Sinkhorn) before picking pseudo-labels, to stop one class from drifting",
    )
    p.add_argument("--pseudo-knn", type=int, default=0, help="kNN-smooth test probabilities before picking pseudo-labels")
    # head training
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--anchor", type=float, default=0.0,
        help="Extra L2 pull of the trained head towards its init (0 = rely on init + few steps; >=0.01 barely moves)",
    )
    # test-time post-processing
    p.add_argument("--knn", type=int, default=0)
    p.add_argument("--knn-weight", type=float, default=0.5)
    p.add_argument("--knn-iters", type=int, default=1, help=">1 = iterative label propagation on the kNN graph")
    p.add_argument("--balance", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_split(root: Path, split: str, device) -> torch.Tensor:
    x = torch.from_numpy(np.load(root / f"{split}_features.npy").astype(np.float32)).to(device)
    return F.normalize(x, dim=-1)


def accuracy(head: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> float:
    return float(((x @ head).argmax(1) == y).float().mean())


def train_head(x, y, init, steps, lr, anchor, weights=None) -> torch.Tensor:
    """Full-batch cross-entropy on normalized features, logits = SCALE * x @ W.

    The L2 term pulls W towards its initialisation, which keeps the head close to the
    (robust) zero-shot solution instead of overfitting the training distribution.
    """
    w0 = init.detach().clone()
    w = init.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([w], lr=lr)
    for _ in range(steps):
        loss = F.cross_entropy(SCALE * x @ w, y, reduction="none")
        loss = (loss * weights).sum() / weights.sum() if weights is not None else loss.mean()
        loss = loss + anchor * (w - w0).pow(2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return w.detach()


def select_pseudo_labels(probs: torch.Tensor, frac: float):
    """Per predicted class, keep the `frac` most confident test samples (keeps classes balanced)."""
    conf, pred = probs.max(1)
    keep = torch.zeros_like(pred, dtype=torch.bool)
    for c in range(probs.shape[1]):
        idx = (pred == c).nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            continue
        k = max(1, int(round(frac * len(idx))))
        keep[idx[conf[idx].topk(k).indices]] = True
    return keep, pred


@torch.no_grad()
def knn_graph(x: torch.Tensor, k: int, chunk: int = 2048) -> torch.Tensor:
    """Indices of the k nearest neighbours of every row (cosine similarity, self excluded)."""
    idx = torch.empty(x.shape[0], k, dtype=torch.long, device=x.device)
    for s in range(0, x.shape[0], chunk):
        idx[s : s + chunk] = (x[s : s + chunk] @ x.T).topk(k + 1, dim=1).indices[:, 1:]
    return idx


@torch.no_grad()
def knn_smooth(x: torch.Tensor, probs: torch.Tensor, k: int, weight: float, iters: int = 1) -> torch.Tensor:
    """Label propagation on the test kNN graph.

    iters=1 is a single smoothing step: (1 - w) * P0 + w * mean of neighbours' P0.
    iters>1 repeats F <- (1 - w) * P0 + w * mean of neighbours' F (Zhou et al., 2004),
    which lets labels spread further along dense regions of the test distribution.
    """
    idx = knn_graph(x, k)
    f = probs
    for _ in range(iters):
        f = (1 - weight) * probs + weight * f[idx].mean(1)
    return f


@torch.no_grad()
def sinkhorn_balance(probs: torch.Tensor, iters: int = 20) -> torch.Tensor:
    q = probs.clone()
    n, k = q.shape
    for _ in range(iters):
        q = q / q.sum(0, keepdim=True) * (n / k)
        q = q / q.sum(1, keepdim=True)
    return q


def distribution(pred: torch.Tensor) -> dict:
    counts = torch.bincount(pred.cpu(), minlength=len(CLASS_NAMES)).numpy()
    return {name: round(float(c) / len(pred), 4) for name, c in zip(CLASS_NAMES, counts)}


def ensemble_probs(heads, xs, weights) -> torch.Tensor:
    """Weighted average of per-model softmax probabilities."""
    return sum(w * (SCALE * x @ h).softmax(1) for h, x, w in zip(heads, xs, weights))


def ensemble_acc(heads, xs, weights, y) -> float:
    return float((ensemble_probs(heads, xs, weights).argmax(1) == y).float().mean())


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)
    roots = args.features
    names = [r.name for r in roots]
    n_models = len(roots)
    weights = args.model_weights or [1.0] * n_models
    if len(weights) != n_models:
        raise ValueError("--model-weights needs one value per --features directory")
    weights = [w / sum(weights) for w in weights]
    report: dict = {"config": {k: str(v) for k, v in vars(args).items()}}

    # Load every model's features. All directories must describe the same images in the same order.
    x_val = [load_split(r, "validation", device) for r in roots]
    x_test = [load_split(r, "test", device) for r in roots]
    y_val = torch.from_numpy(np.load(roots[0] / "validation_labels.npy")).to(device)
    test_ids = np.load(roots[0] / "test_ids.npy")
    for r in roots[1:]:
        if not np.array_equal(np.load(r / "test_ids.npy"), test_ids):
            raise ValueError(f"{r}: test ids differ from {roots[0]}")
        if not np.array_equal(np.load(r / "validation_labels.npy"), y_val.cpu().numpy()):
            raise ValueError(f"{r}: validation labels differ from {roots[0]}")
    # Joint feature space for kNN: concatenation of the per-model normalized features.
    x_test_joint = F.normalize(torch.cat(x_test, dim=1), dim=-1)

    # 1. zero-shot heads
    zs_heads = [
        F.normalize(torch.from_numpy(np.load(r / "heads.npz")[args.head]).float().to(device), dim=0) for r in roots
    ]
    single_preds = [(x @ h).argmax(1) for x, h in zip(x_test, zs_heads)]
    for name, h, xv, pred in zip(names, zs_heads, x_val, single_preds):
        acc = accuracy(h, xv, y_val)
        report[f"val_acc_zeroshot_{name}"] = acc
        print(f"[zero-shot {name}] val acc {acc:.4f}", flush=True)
    for name, h, xt in zip(names, zs_heads, x_test):
        conf = float((SCALE * xt @ h).softmax(1).max(1).values.mean())
        report[f"test_mean_confidence_{name}"] = conf
        print(f"[zero-shot {name}] mean test confidence {conf:.3f}", flush=True)
    for i in range(n_models):
        for j in range(i + 1, n_models):
            agree = float((single_preds[i] == single_preds[j]).float().mean())
            report[f"test_agreement_{names[i]}_{names[j]}"] = agree
            print(f"[zero-shot] test agreement between {names[i]} and {names[j]}: {agree:.2%}", flush=True)
    zs_pred = ensemble_probs(zs_heads, x_test, weights).argmax(1)
    report["val_acc_zeroshot"] = ensemble_acc(zs_heads, x_val, weights, y_val)
    report["test_changed_vs_first_model"] = float((zs_pred != single_preds[0]).float().mean())
    if n_models > 1:
        print(
            f"[zero-shot ensemble w={[round(w, 2) for w in weights]}] val acc {report['val_acc_zeroshot']:.4f}, "
            f"test preds changed vs {names[0]} alone {report['test_changed_vs_first_model']:.2%}",
            flush=True,
        )
    heads = list(zs_heads)

    # 2. linear probe on CIFAR train + WiSE interpolation (per model)
    x_train = y_train = None
    if args.wise_alpha > 0 or args.pseudo_with_train:
        x_train = [load_split(r, "train", device) for r in roots]
        y_train = torch.from_numpy(np.load(roots[0] / "train_labels.npy")).to(device)
    if args.wise_alpha > 0:
        lp_heads = [train_head(x, y_train, h, args.steps, args.lr, args.anchor) for x, h in zip(x_train, zs_heads)]
        heads = [(1 - args.wise_alpha) * h0 + args.wise_alpha * h1 for h0, h1 in zip(zs_heads, lp_heads)]
        report["val_acc_linear_probe"] = ensemble_acc(lp_heads, x_val, weights, y_val)
        report["val_acc_wise"] = ensemble_acc(heads, x_val, weights, y_val)
        test_pred = ensemble_probs(heads, x_test, weights).argmax(1)
        report["test_changed_vs_zeroshot_after_wise"] = float((test_pred != zs_pred).float().mean())
        print(
            f"[wise a={args.wise_alpha}] val acc probe {report['val_acc_linear_probe']:.4f}, "
            f"wise {report['val_acc_wise']:.4f}, test preds changed {report['test_changed_vs_zeroshot_after_wise']:.2%}",
            flush=True,
        )

    # 3. transductive pseudo-labelling: ensemble picks the labels, each model's head is refit on its own features
    for r in range(1, args.pseudo_rounds + 1):
        probs = ensemble_probs(heads, x_test, weights)
        if args.pseudo_knn > 0:
            probs = knn_smooth(x_test_joint, probs, args.pseudo_knn, args.knn_weight)
        if args.pseudo_balance:
            probs = sinkhorn_balance(probs)
        keep, pred = select_pseudo_labels(probs, args.pseudo_frac)
        new_heads = []
        for m in range(n_models):
            xs, ys = x_test[m][keep], pred[keep]
            if args.pseudo_with_train:
                xs, ys = torch.cat([xs, x_train[m]]), torch.cat([ys, y_train])
            new_heads.append(train_head(xs, ys, heads[m], args.steps, args.lr, args.anchor))
        heads = new_heads
        new_pred = ensemble_probs(heads, x_test, weights).argmax(1)
        changed = float((new_pred != zs_pred).float().mean())
        val_acc = ensemble_acc(heads, x_val, weights, y_val)
        dist = distribution(new_pred)
        report[f"pseudo_round_{r}"] = {
            "kept": int(keep.sum()), "val_acc": val_acc, "test_changed_vs_zeroshot": changed, "test_distribution": dist,
        }
        print(
            f"[pseudo round {r}] kept {int(keep.sum())} test samples, val acc {val_acc:.4f}, "
            f"test preds changed vs zero-shot {changed:.2%}, class share min/max "
            f"{min(dist.values()):.1%}/{max(dist.values()):.1%}",
            flush=True,
        )

    # 4-5. test-time post-processing
    probs = ensemble_probs(heads, x_test, weights)
    if args.knn > 0:
        probs = knn_smooth(x_test_joint, probs, args.knn, args.knn_weight, args.knn_iters)
    if args.balance:
        probs = sinkhorn_balance(probs)
    final_pred = probs.argmax(1)

    report["test_changed_vs_zeroshot"] = float((final_pred != zs_pred).float().mean())
    report["test_changed_vs_first_model_zeroshot"] = float((final_pred != single_preds[0]).float().mean())
    report["test_mean_confidence"] = float(probs.max(1).values.mean())
    report["test_distribution"] = distribution(final_pred)
    print(f"[final] test preds changed vs zero-shot ensemble: {report['test_changed_vs_zeroshot']:.2%}")
    print(f"[final] test preds changed vs {names[0]} zero-shot alone: {report['test_changed_vs_first_model_zeroshot']:.2%}")
    print("[final] predicted class distribution on test:")
    for name, frac in report["test_distribution"].items():
        print(f"  {name:<11}{frac:.1%}")

    sample = pd.read_csv(args.data / "sample_submission.csv")
    if not np.array_equal(sample["id"].to_numpy(), test_ids):
        raise ValueError("Test ids are not aligned with sample_submission.csv.")
    pd.DataFrame({"id": test_ids, "label": final_pred.cpu().numpy()}).to_csv(args.output / "submission.csv", index=False)
    np.save(args.output / "test_probs.npy", probs.cpu().numpy())
    np.savez(args.output / "heads.npz", **{n: h.cpu().numpy() for n, h in zip(names, heads)})
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(f"saved {args.output / 'submission.csv'}")


if __name__ == "__main__":
    main()
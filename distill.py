"""Noisy-Student distillation on the unlabeled test images.

A fixed teacher (the probabilities saved by feature_experiments.py, e.g. the
3-model ensemble with kNN + balance) provides soft targets for every test image.
A CLIP image encoder (the student) is fine-tuned end to end to match them under
strong augmentations, so that its backbone learns test-domain features instead
of only re-drawing a decision boundary in a frozen feature space.

No test label is used: only test images and the teacher's own predictions.
CIFAR-10 validation accuracy is reported each epoch as a sanity check.

The student's head starts from the zero-shot text embeddings. At the end the
script writes a submission, and a features directory in the same format as
extract_features.py, so the student can be added to the ensemble:

    python feature_experiments.py --features $B $D $S runs/distill_l14/features --knn 10 --balance ...

Training checkpoints are saved every epoch; relaunching the same command resumes.

Example:
    python distill.py --teacher runs/ens3_knn_bal/test_probs.npy --output runs/distill_l14 \
        --model ViT-L-14 --pretrained laion2b_s32b_b82k --epochs 4 --batch-size 64
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from clip_zeroshot import build_classifier, get_templates
from data import CLASS_NAMES

SCALE = 100.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--teacher", type=Path, required=True, help="(N_test, 10) probabilities, e.g. test_probs.npy")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default="ViT-L-14")
    p.add_argument("--pretrained", default="laion2b_s32b_b82k")
    p.add_argument("--head", default="openai_imagenet", help="Prompt set used to initialise the student head")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-5, help="Backbone learning rate")
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup", type=float, default=0.05, help="Fraction of steps used for linear warmup")
    p.add_argument("--temperature", type=float, default=1.0, help="<1 sharpens the teacher targets")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--grad-checkpointing", action="store_true", help="Saves memory, ~30%% slower")
    p.add_argument("--randaugment", action="store_true", help="Add RandAugment (2 ops, magnitude 9) at 32x32")
    p.add_argument("--mixup", type=float, default=0.0, help="Mixup Beta(a, a) parameter, 0 = off")
    p.add_argument("--cutmix", type=float, default=0.0, help="CutMix Beta(a, a) parameter, 0 = off")
    p.add_argument("--mix-prob", type=float, default=0.8, help="Probability of applying mixup/cutmix to a batch")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


class NpzImages(Dataset):
    """Returns (transformed image, row index) so that targets can be looked up by index."""

    def __init__(self, path: Path, transform):
        with np.load(path) as arrays:
            self.images = arrays["images"]
            self.labels = arrays["labels"] if "labels" in arrays.files else None
            self.ids = arrays["ids"] if "ids" in arrays.files else None
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i):
        return self.transform(Image.fromarray(self.images[i])), i


class Student(torch.nn.Module):
    """CLIP image encoder + cosine classifier initialised from the text embeddings."""

    def __init__(self, visual: torch.nn.Module, head_init: torch.Tensor):
        super().__init__()
        self.visual = visual
        self.head = torch.nn.Parameter(head_init.clone())  # (dim, 10)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.visual(x).float(), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return SCALE * self.features(x) @ F.normalize(self.head, dim=0)


def mix_batch(images: torch.Tensor, targets: torch.Tensor, args):
    """Mixup or CutMix on a batch, with the teacher targets mixed in the same proportion."""
    use_mixup, use_cutmix = args.mixup > 0, args.cutmix > 0
    if not (use_mixup or use_cutmix) or np.random.rand() > args.mix_prob:
        return images, targets
    perm = torch.randperm(images.shape[0], device=images.device)
    if use_cutmix and (not use_mixup or np.random.rand() < 0.5):
        lam = np.random.beta(args.cutmix, args.cutmix)
        h, w = images.shape[-2:]
        cut_h, cut_w = int(h * math.sqrt(1 - lam)), int(w * math.sqrt(1 - lam))
        cy, cx = np.random.randint(h), np.random.randint(w)
        y0, y1 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, h)
        x0, x1 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, w)
        images = images.clone()
        images[:, :, y0:y1, x0:x1] = images[perm, :, y0:y1, x0:x1]
        lam = 1 - (y1 - y0) * (x1 - x0) / (h * w)  # exact area kept from the original image
    else:
        lam = np.random.beta(args.mixup, args.mixup)
        images = lam * images + (1 - lam) * images[perm]
    return images, lam * targets + (1 - lam) * targets[perm]


def cosine_lr(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def run_eval(student, loader, device, amp_dtype, hflip_tta=True):
    """Returns normalized features (with optional hflip averaging) and logits."""
    student.eval()
    feats = []
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            f = student.features(images)
            if hflip_tta:
                f = F.normalize(f + student.features(images.flip(-1)), dim=-1)
        feats.append(f.float().cpu())
    feats = torch.cat(feats)
    logits = SCALE * feats @ F.normalize(student.head.detach().float().cpu(), dim=0)
    return feats, logits


def main() -> None:
    import open_clip

    args = parse_args()
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and not use_bf16)

    # Model and zero-shot head
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model)
    model.to(device)
    head_init = build_classifier(model, tokenizer, get_templates(args.head), device).float()
    student = Student(model.visual, head_init).to(device)
    del model  # drop the text tower
    if args.grad_checkpointing:
        student.visual.set_grad_checkpointing(True)

    # Data. Strong augmentations are applied at native 32x32 resolution, then CLIP preprocessing.
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.6, 1.0), ratio=(0.8, 1.25)),
        transforms.RandomHorizontalFlip(),
        *([transforms.RandAugment(num_ops=2, magnitude=9)] if args.randaugment else []),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        preprocess,
    ])
    test_train = NpzImages(args.data / "test.npz", train_tf)
    test_eval = NpzImages(args.data / "test.npz", preprocess)
    val_eval = NpzImages(args.data / "validation.npz", preprocess)

    teacher = torch.from_numpy(np.load(args.teacher)).float()
    if teacher.shape != (len(test_train), len(CLASS_NAMES)):
        raise ValueError(f"teacher has shape {tuple(teacher.shape)}, expected ({len(test_train)}, 10)")
    if args.temperature != 1.0:
        teacher = F.normalize(teacher.clamp_min(1e-8) ** (1 / args.temperature), p=1, dim=1)
    teacher = teacher.to(device)
    val_labels = torch.from_numpy(val_eval.labels.astype(np.int64))

    loader_kw = dict(num_workers=args.workers, pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    train_loader = DataLoader(test_train, batch_size=args.batch_size, shuffle=True, drop_last=True, **loader_kw)
    eval_kw = dict(batch_size=args.batch_size * 2, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_eval, **eval_kw)
    val_loader = DataLoader(val_eval, **eval_kw)

    optimizer = torch.optim.AdamW(
        [
            {"params": student.visual.parameters(), "lr": args.lr},
            {"params": [student.head], "lr": args.head_lr, "weight_decay": 0.0},
        ],
        weight_decay=args.weight_decay,
    )
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(args.warmup * total_steps)

    # Resume
    ckpt_path = args.output / "checkpoint.pt"
    start_epoch, step, history = 0, 0, []
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        student.load_state_dict(ckpt["student"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch, step, history = ckpt["epoch"], ckpt["step"], ckpt["history"]
        print(f"resumed from epoch {start_epoch}", flush=True)

    print(
        f"device={device} amp={amp_dtype} model={args.model}/{args.pretrained} "
        f"test images={len(test_train)} steps/epoch={len(train_loader)} epochs={args.epochs}",
        flush=True,
    )
    if start_epoch == 0:
        _, val_logits = run_eval(student, val_loader, device, amp_dtype, hflip_tta=False)
        print(f"[epoch 0] CIFAR val acc (zero-shot student) {(val_logits.argmax(1) == val_labels).float().mean():.4f}")

    for epoch in range(start_epoch, args.epochs):
        student.train()
        t0, running = time.time(), 0.0
        for i, (images, idx) in enumerate(train_loader, 1):
            for g, base in zip(optimizer.param_groups, base_lrs):
                g["lr"] = base * cosine_lr(step, total_steps, warmup_steps)
            images = images.to(device, non_blocking=True)
            images, targets = mix_batch(images, teacher[idx.to(device)], args)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                logits = student(images)
            loss = -(targets * F.log_softmax(logits.float(), dim=1)).sum(1).mean()
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            step += 1
            running += loss.item()
            if i % 100 == 0 or i == len(train_loader):
                print(
                    f"[epoch {epoch + 1}] step {i}/{len(train_loader)} loss {running / i:.4f} "
                    f"{(time.time() - t0) / 60:.1f} min",
                    flush=True,
                )

        _, val_logits = run_eval(student, val_loader, device, amp_dtype, hflip_tta=False)
        val_acc = float((val_logits.argmax(1) == val_labels).float().mean())
        _, test_logits = run_eval(student, test_loader, device, amp_dtype, hflip_tta=False)
        agree = float((test_logits.argmax(1) == teacher.argmax(1).cpu()).float().mean())
        history.append({"epoch": epoch + 1, "loss": running / len(train_loader), "val_acc": val_acc, "teacher_agreement": agree})
        print(f"[epoch {epoch + 1}] CIFAR val acc {val_acc:.4f}, test agreement with teacher {agree:.2%}", flush=True)
        torch.save(
            {"student": student.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch + 1, "step": step, "history": history},
            ckpt_path,
        )

    # Final predictions with hflip TTA, plus a features directory for feature_experiments.py
    val_feats, val_logits = run_eval(student, val_loader, device, amp_dtype)
    test_feats, test_logits = run_eval(student, test_loader, device, amp_dtype)
    val_acc = float((val_logits.argmax(1) == val_labels).float().mean())
    pred = test_logits.argmax(1)
    agree = float((pred == teacher.argmax(1).cpu()).float().mean())
    print(f"[final, hflip TTA] CIFAR val acc {val_acc:.4f}, test agreement with teacher {agree:.2%}")

    feat_dir = args.output / "features"
    feat_dir.mkdir(exist_ok=True)
    np.save(feat_dir / "validation_features.npy", val_feats.numpy().astype(np.float16))
    np.save(feat_dir / "validation_labels.npy", val_eval.labels.astype(np.int64))
    np.save(feat_dir / "test_features.npy", test_feats.numpy().astype(np.float16))
    np.save(feat_dir / "test_ids.npy", test_eval.ids.astype(np.int64))
    head = F.normalize(student.head.detach().float().cpu(), dim=0).numpy()
    np.savez(feat_dir / "heads.npz", **{args.head: head})

    sample = pd.read_csv(args.data / "sample_submission.csv")
    if not np.array_equal(sample["id"].to_numpy(), test_eval.ids):
        raise ValueError("Test ids are not aligned with sample_submission.csv.")
    pd.DataFrame({"id": test_eval.ids, "label": pred.numpy()}).to_csv(args.output / "submission.csv", index=False)
    np.save(args.output / "test_probs.npy", test_logits.softmax(1).numpy())
    counts = torch.bincount(pred, minlength=len(CLASS_NAMES)).numpy()
    report = {
        "config": {k: str(v) for k, v in vars(args).items()},
        "history": history,
        "final_val_acc": val_acc,
        "final_teacher_agreement": agree,
        "test_distribution": {n: round(float(c) / len(pred), 4) for n, c in zip(CLASS_NAMES, counts)},
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(f"saved {args.output / 'submission.csv'} and {feat_dir}")


if __name__ == "__main__":
    main()
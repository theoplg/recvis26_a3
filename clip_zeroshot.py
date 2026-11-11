"""Zero-shot CLIP classification with optional prompt ensembling.

Evaluates on the labeled CIFAR-10 validation set, then predicts the test set and
writes submission.csv plus the raw test logits (useful later for ensembling or
pseudo-labelling).

Example:
    python3 clip_zeroshot.py --data data --output runs/clip_b32 \
        --model ViT-B-32 --pretrained laion2b_s34b_b79k --prompts ensemble
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import CLASS_NAMES, NpzImageDataset

# Templates chosen to cover the styles seen in the test domain
# (toys, sketches, paintings, 3D renders, pixel art, grayscale...).
ENSEMBLE_TEMPLATES = (
    "a photo of a {}.",
    "a blurry photo of a {}.",
    "a low resolution photo of a {}.",
    "a black and white photo of a {}.",
    "a drawing of a {}.",
    "a sketch of a {}.",
    "a pencil sketch of a {}.",
    "a painting of a {}.",
    "a cartoon {}.",
    "a 3d render of a {}.",
    "a toy {}.",
    "a plastic {}.",
    "a wooden {}.",
    "a sculpture of a {}.",
    "an origami {}.",
    "pixel art of a {}.",
    "a mosaic of a {}.",
    "a rendition of a {}.",
)
SIMPLE_TEMPLATES = ("a photo of a {}.",)

# The 18 CIFAR-10 templates published with CLIP (Radford et al., 2021,
# github.com/openai/CLIP, data/prompts.md). Chosen without any knowledge of
# this competition's test domain.
OPENAI_CIFAR10_TEMPLATES = (
    "a photo of a {}.",
    "a blurry photo of a {}.",
    "a black and white photo of a {}.",
    "a low contrast photo of a {}.",
    "a high contrast photo of a {}.",
    "a bad photo of a {}.",
    "a good photo of a {}.",
    "a photo of a small {}.",
    "a photo of a big {}.",
    "a photo of the {}.",
    "a blurry photo of the {}.",
    "a black and white photo of the {}.",
    "a low contrast photo of the {}.",
    "a high contrast photo of the {}.",
    "a bad photo of the {}.",
    "a good photo of the {}.",
    "a photo of the small {}.",
    "a photo of the big {}.",
)


def get_templates(name: str) -> tuple[str, ...]:
    if name == "simple":
        return SIMPLE_TEMPLATES
    if name == "ensemble":
        return ENSEMBLE_TEMPLATES
    if name == "openai_cifar10":
        return OPENAI_CIFAR10_TEMPLATES
    if name == "openai_imagenet":
        # The 80 ImageNet templates from the CLIP paper, shipped with open_clip
        # as lambdas. Calling them with "{}" turns them back into format strings.
        from open_clip import OPENAI_IMAGENET_TEMPLATES

        return tuple(template("{}") for template in OPENAI_IMAGENET_TEMPLATES)
    raise ValueError(f"Unknown prompt set: {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    parser.add_argument(
        "--prompts",
        choices=("simple", "ensemble", "openai_cifar10", "openai_imagenet"),
        default="ensemble",
        help="simple: 1 prompt | ensemble: 18 domain-inspired prompts | "
        "openai_cifar10: 18 CLIP-paper CIFAR-10 prompts | openai_imagenet: 80 CLIP-paper ImageNet prompts",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Run the model in half precision (halves memory, ~2x faster). Recommended for ViT-H/g/bigG.",
    )
    return parser.parse_args()


@torch.inference_mode()
def build_classifier(model, tokenizer, templates, device) -> torch.Tensor:
    """One normalized text embedding per class, averaged over templates."""
    weights = []
    for name in CLASS_NAMES:
        tokens = tokenizer([t.format(name) for t in templates]).to(device)
        emb = model.encode_text(tokens).float()
        emb = emb / emb.norm(dim=-1, keepdim=True)
        emb = emb.mean(dim=0)
        weights.append(emb / emb.norm())
    return torch.stack(weights, dim=1)  # (dim, 10), float32


@torch.inference_mode()
def compute_logits(model, classifier, dataset, batch_size, workers, device, dtype, name=""):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    all_logits, all_second = [], []
    start, n_batches = time.time(), len(loader)
    for step, (images, second) in enumerate(loader, 1):
        feats = model.encode_image(images.to(device, dtype=dtype)).float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
        all_logits.append((100.0 * feats @ classifier).cpu())
        all_second.append(second)
        if step % 20 == 0 or step == n_batches:
            elapsed = time.time() - start
            print(f"  [{name}] batch {step}/{n_batches}  {elapsed / 60:.1f} min", flush=True)
    return torch.cat(all_logits).numpy(), torch.cat(all_second).numpy()


def main() -> None:
    import open_clip

    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    dtype = torch.float16 if args.fp16 and device.type == "cuda" else torch.float32
    model.to(device=device, dtype=dtype).eval()
    tokenizer = open_clip.get_tokenizer(args.model)
    templates = get_templates(args.prompts)
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(
        f"device={device} model={args.model} pretrained={args.pretrained} "
        f"params={n_params:.2f}B dtype={dtype} prompts={args.prompts} templates={len(templates)}",
        flush=True,
    )

    classifier = build_classifier(model, tokenizer, templates, device)

    val_set = NpzImageDataset(args.data / "validation.npz", transform=preprocess, require_labels=True)
    val_logits, val_labels = compute_logits(
        model, classifier, val_set, args.batch_size, args.workers, device, dtype, "val"
    )
    val_acc = float((val_logits.argmax(1) == val_labels).mean())
    print(f"CIFAR-10 validation accuracy: {val_acc:.4f}", flush=True)
    np.save(args.output / "val_logits.npy", val_logits)

    test_set = NpzImageDataset(args.data / "test.npz", transform=preprocess, require_labels=False)
    test_logits, test_ids = compute_logits(
        model, classifier, test_set, args.batch_size, args.workers, device, dtype, "test"
    )
    predictions = test_logits.argmax(1)

    sample = pd.read_csv(args.data / "sample_submission.csv")
    if not np.array_equal(sample["id"].to_numpy(), test_ids):
        raise ValueError("Test ids are not aligned with sample_submission.csv.")

    submission = pd.DataFrame({"id": test_ids, "label": predictions})
    submission.to_csv(args.output / "submission.csv", index=False)
    np.save(args.output / "test_logits.npy", test_logits)

    counts = np.bincount(predictions, minlength=len(CLASS_NAMES))
    print("predicted class distribution on test:")
    for name, count in zip(CLASS_NAMES, counts):
        print(f"  {name:<11}{count:>7}  ({count / len(predictions):.1%})")
    print(f"saved {args.output / 'submission.csv'}")
    print(f"saved {args.output / 'test_logits.npy'}")


if __name__ == "__main__":
    main()
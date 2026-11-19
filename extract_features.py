"""Extract frozen CLIP image features for train / validation / test, with resume.

The image encoder runs once over every split. Features are written to disk in
chunks, so an interrupted job can be relaunched with the same command and only
the missing chunks are computed. Text classifier heads (one vector per class)
are saved alongside, so zero-shot, linear-probe and pseudo-label experiments can
then run on CPU in seconds without touching the GPU again.

Output layout (``--output``):
    meta.json                     model, checkpoint, dtype, feature dim, ...
    heads.npz                     zero-shot heads: one (dim, 10) matrix per prompt set
    {split}_features.npy          (N, dim) float16, raw image embeddings (not normalized)
    {split}_labels.npy            (N,) int64, train and validation only
    {split}_ids.npy               (N,) int64, test only
    chunks/{split}/*.npy          intermediate chunks, safe to delete once merged

Example:
    python extract_features.py --output features/bigg \
        --model ViT-bigG-14 --pretrained laion2b_s39b_b160k --fp16 --batch-size 64
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from clip_zeroshot import build_classifier, get_templates
from data import CLASS_NAMES, NpzImageDataset

SPLITS = ("validation", "test", "train")
PROMPT_SETS = ("simple", "openai_imagenet", "ensemble")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="ViT-bigG-14")
    parser.add_argument("--pretrained", default="laion2b_s39b_b160k")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=2048, help="Images per saved chunk.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--fp16", action="store_true", help="Half precision (recommended for large models).")
    return parser.parse_args()


def save_atomic(path: Path, array: np.ndarray) -> None:
    """Write to a temporary file then rename, so a crash never leaves a truncated chunk."""
    tmp = path.with_name(path.name + ".tmp.npy")
    np.save(tmp, array)
    os.replace(tmp, path)


@torch.inference_mode()
def extract_split(model, preprocess, args, split, device, dtype) -> None:
    final_path = args.output / f"{split}_features.npy"
    if final_path.exists():
        print(f"[{split}] already merged, skipping", flush=True)
        return

    require_labels = split != "test"
    dataset = NpzImageDataset(args.data / f"{split}.npz", transform=preprocess, require_labels=require_labels)
    n = len(dataset)
    chunk_dir = args.output / "chunks" / split
    chunk_dir.mkdir(parents=True, exist_ok=True)
    starts = list(range(0, n, args.chunk_size))
    print(f"[{split}] {n} images, {len(starts)} chunks of {args.chunk_size}", flush=True)

    t0, done_now = time.time(), 0
    for idx, start in enumerate(starts):
        chunk_path = chunk_dir / f"{idx:05d}.npy"
        if chunk_path.exists():
            continue
        stop = min(start + args.chunk_size, n)
        loader = DataLoader(
            Subset(dataset, range(start, stop)),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
        )
        feats = []
        for images, _ in loader:
            feats.append(model.encode_image(images.to(device, dtype=dtype)).float().cpu())
        save_atomic(chunk_path, torch.cat(feats).numpy().astype(np.float16))

        done_now += stop - start
        elapsed = time.time() - t0
        remaining = sum(min(s + args.chunk_size, n) - s for j, s in enumerate(starts) if j > idx)
        eta = elapsed / done_now * remaining / 60
        print(
            f"[{split}] chunk {idx + 1}/{len(starts)} done  "
            f"{elapsed / 60:.1f} min elapsed, ~{eta:.0f} min left for this split",
            flush=True,
        )

    # Merge chunks and save labels / ids in dataset order.
    features = np.concatenate([np.load(chunk_dir / f"{i:05d}.npy") for i in range(len(starts))])
    if features.shape[0] != n:
        raise RuntimeError(f"[{split}] expected {n} features, got {features.shape[0]}")
    if require_labels:
        np.save(args.output / f"{split}_labels.npy", dataset.labels.astype(np.int64))
    else:
        np.save(args.output / f"{split}_ids.npy", dataset.ids.astype(np.int64))
    save_atomic(final_path, features)
    print(f"[{split}] saved {final_path}  shape={features.shape}", flush=True)


def main() -> None:
    import open_clip

    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.fp16 and device.type == "cuda" else torch.float32

    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    model.to(device=device, dtype=dtype).eval()
    tokenizer = open_clip.get_tokenizer(args.model)
    print(f"device={device} model={args.model} pretrained={args.pretrained} dtype={dtype}", flush=True)

    heads_path = args.output / "heads.npz"
    if not heads_path.exists():
        heads = {
            name: build_classifier(model, tokenizer, get_templates(name), device).cpu().numpy()
            for name in PROMPT_SETS
        }
        np.savez(heads_path, **heads)
        print(f"saved {heads_path} ({', '.join(PROMPT_SETS)})", flush=True)

    meta = {
        "model": args.model,
        "pretrained": args.pretrained,
        "dtype": str(dtype),
        "feature_dim": int(np.load(heads_path)[PROMPT_SETS[0]].shape[0]),
        "logit_scale": float(model.logit_scale.exp().item()),
        "class_names": list(CLASS_NAMES),
        "features_normalized": False,
    }
    (args.output / "meta.json").write_text(json.dumps(meta, indent=2))

    for split in args.splits:
        extract_split(model, preprocess, args, split, device, dtype)
    print("all requested splits done", flush=True)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Embed a memmap-backed SKATR dataset with a pretrained ViT backbone and write
summary NPZ files compatible with create_memmap_npz.py.

Input directory must contain:
  - lightcone_*.npz
  - images.memmap
  - labels.memmap

Output directory will contain:
  - lightcone_*.npz with image=(D,) summary vector and label unchanged
"""

import argparse
import glob
import os
import sys
import numpy as np
import torch

from tensordict import MemoryMappedTensor
sys.path.append('/pfs/10/work/hd_pt254-skatr/skatr')
from src.networks.vit import PretrainedViT
from src.utils.transforms import AddSingletonChannel, ShiftMedian, Clamp, Center


def parse_args():
    p = argparse.ArgumentParser(description="Embed dataset using pretrained SKATR backbone")
    p.add_argument("--input_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--backbone_dir", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--median", type=float, default=-2.0)
    p.add_argument("--use_amp", action="store_true")
    # Backbones pretrained with a different input normalisation need that same
    # normalisation at embedding time. 'symlog' (default) is the paper
    # convention, ShiftMedian(median) + Clamp. 'center' is the affine map
    # (x - lo)/(hi - lo) used by the EMA/z-mask pretraining run.
    p.add_argument("--preproc", choices=["symlog", "center"], default="symlog")
    p.add_argument("--center_lo", type=float, default=-120.0)
    p.add_argument("--center_hi", type=float, default=-1.0)
    return p.parse_args()


def main():
    args = parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, "lightcone_*.npz")))
    if len(files) == 0:
        raise ValueError(f"No lightcone_*.npz found in {args.input_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    sample = np.load(files[0])
    image_shape = sample["image"].shape
    label_shape = sample["label"].shape
    n = len(files)

    images_mmap = MemoryMappedTensor.from_filename(
        filename=os.path.join(args.input_dir, "images.memmap"),
        dtype=torch.float32,
        shape=(n, *image_shape),
    )
    labels_mmap = MemoryMappedTensor.from_filename(
        filename=os.path.join(args.input_dir, "labels.memmap"),
        dtype=torch.float32,
        shape=(n, *label_shape),
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = PretrainedViT(
        backbone_dir=args.backbone_dir,
        frozen=True,
        drop_head=True,
        use_head=False,
        interp_pos_encoding=False,
    ).to(device)
    model.eval()

    if args.preproc == "center":
        x_transforms = [AddSingletonChannel(), Center(args.center_lo, args.center_hi)]
    else:
        x_transforms = [AddSingletonChannel(), ShiftMedian(args.median), Clamp()]

    print("=" * 72)
    print("EMBED DATASET WITH PRETRAINED SKATR")
    print("=" * 72)
    print(f"Input dir:      {args.input_dir}")
    print(f"Output dir:     {args.output_dir}")
    print(f"Backbone dir:   {args.backbone_dir}")
    print(f"Num samples:    {n}")
    print(f"Image shape in: {image_shape}")
    print(f"Label shape:    {label_shape}")
    print(f"Device:         {device}")

    with torch.inference_mode():
        for start in range(0, n, args.batch_size):
            end = min(start + args.batch_size, n)

            batch = images_mmap[start:end]
            for t in x_transforms:
                batch = t.forward(batch)
            batch = batch.to(device)

            with torch.autocast(device.type, enabled=args.use_amp):
                summaries = model(batch)

            # ViT without head returns (B, T, D) -> mean-pool over tokens
            if summaries.ndim == 3:
                summaries = summaries.mean(1)

            summaries = summaries.float().cpu().numpy()
            labels = labels_mmap[start:end].cpu().numpy().astype(np.float32)

            for i, idx in enumerate(range(start, end)):
                out_name = os.path.basename(files[idx])
                out_path = os.path.join(args.output_dir, out_name)
                np.savez(
                    out_path,
                    image=summaries[i].astype(np.float32),
                    label=labels[i],
                )

            if start == 0:
                print(f"Summary dim: {summaries.shape[1]}")
            print(f"Embedded {end}/{n}", end="\r", flush=True)

    print("\nDone embedding.")


if __name__ == "__main__":
    main()

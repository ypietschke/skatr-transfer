#!/usr/bin/env python3
"""
Same as embed_dataset_memmap.py but writes the full D4 orbit of summaries
per lightcone (8 transforms: identity, 3 rotations, 4 reflected variants).

Output NPZ format mirrors the upstream summary format but with
  image: (8, D) instead of (D,)
where image[0] is the identity (so existing single-summary loaders that
read image[0] continue to work).

Stage 0 use: run this once per (backbone, source lightcone dir) pair.
Stage 3 use: run this for every backbone whose CFM ensembles you want
to redo with cfm_aug=True.

CLI matches embed_dataset_memmap.py so the upstream submit scripts stay
familiar; the only extra knob is --include_identity (defaults to True).
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
from src.utils.transforms import AddSingletonChannel, ShiftMedian, Clamp
from src.utils.augmentations import RotateAndReflect


def parse_args():
    p = argparse.ArgumentParser(description="Embed dataset using pretrained SKATR backbone, writing the full D4 orbit per lightcone")
    p.add_argument("--input_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--backbone_dir", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--median", type=float, default=-2.0)
    p.add_argument("--use_amp", action="store_true")
    p.add_argument("--include_identity", action="store_true", default=True,
                   help="Include identity as slot 0 (default True; needed for downstream eval which uses slot 0).")
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

    x_transforms = [AddSingletonChannel(), ShiftMedian(args.median), Clamp()]

    # RotateAndReflect: (i,j) where i in {0,1} = reflection, j in {0,1,2,3} = 90deg rotation.
    # idcs[0] is (0,0) = identity when include_identity=True. We rely on slot 0 = identity
    # downstream (val/test branch of the CFM dataset uses slot 0 for deterministic eval).
    aug = RotateAndReflect(include_identity=args.include_identity)
    n_aug = len(aug.idcs)
    assert aug.idcs[0] == (0, 0), (
        "Expected idcs[0] == (0,0) identity for downstream slot-0-eval convention; "
        f"got {aug.idcs[0]}. RotateAndReflect implementation changed."
    )

    print("=" * 72)
    print("EMBED DATASET WITH PRETRAINED SKATR (D4 augmented)")
    print("=" * 72)
    print(f"Input dir:      {args.input_dir}")
    print(f"Output dir:     {args.output_dir}")
    print(f"Backbone dir:   {args.backbone_dir}")
    print(f"Num samples:    {n}")
    print(f"Image shape in: {image_shape}")
    print(f"Label shape:    {label_shape}")
    print(f"D4 orbit size:  {n_aug}  (include_identity={args.include_identity})")
    print(f"Device:         {device}")

    # Also write the (rot_idx, ref_idx) mapping next to the npz files so it is
    # recoverable later if someone wants to invert / interpret slots.
    with open(os.path.join(args.output_dir, "aug_index.txt"), "w") as f:
        f.write("# Slot -> (ref_idx, rot_idx) for D4 transformations applied per lightcone.\n")
        f.write("# Slot 0 is the identity and is what cfm_aug=False / sample.py read.\n")
        for slot, (ref_idx, rot_idx) in enumerate(aug.idcs):
            f.write(f"{slot}\t{ref_idx}\t{rot_idx}\n")

    with torch.inference_mode():
        for start in range(0, n, args.batch_size):
            end = min(start + args.batch_size, n)
            B = end - start

            batch = images_mmap[start:end]
            for t in x_transforms:
                batch = t.forward(batch)
            batch = batch.to(device)
            # batch shape: (B, 1, x, y, z) after AddSingletonChannel

            # enumerate returns list of n_aug tensors, each (B, 1, x, y, z)
            xs = aug.enumerate(batch)
            assert len(xs) == n_aug

            # encode each transform, mean-pool tokens
            summaries_per_slot = []
            for xa in xs:
                with torch.autocast(device.type, enabled=args.use_amp):
                    s = model(xa)
                if s.ndim == 3:
                    s = s.mean(1)
                summaries_per_slot.append(s.float().cpu().numpy())  # (B, D)

            # stack to (B, n_aug, D)
            summaries_stack = np.stack(summaries_per_slot, axis=1).astype(np.float32)
            labels = labels_mmap[start:end].cpu().numpy().astype(np.float32)

            for i, idx in enumerate(range(start, end)):
                out_name = os.path.basename(files[idx])
                out_path = os.path.join(args.output_dir, out_name)
                np.savez(
                    out_path,
                    image=summaries_stack[i],   # (n_aug, D)
                    label=labels[i],
                )

            if start == 0:
                print(f"Summary shape per lightcone: {summaries_stack[0].shape}")
            print(f"Embedded {end}/{n}", end="\r", flush=True)

    print("\nDone embedding (D4-augmented).")


if __name__ == "__main__":
    main()

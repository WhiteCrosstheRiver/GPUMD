#!/usr/bin/env python3
"""Deterministic geometry-deduplicated train/validation/test split of real XYZ."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from reference import read_xyz


def identity(frame):
    return (
        tuple(frame.species),
        np.round(frame.cell, 9).tobytes(),
        np.round(frame.positions, 9).tobytes(),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20261014)
    p.add_argument(
        "--preserve-training",
        action="store_true",
        help="Keep first source as training; split remaining unique frames equally for validation/test",
    )
    a = p.parse_args()
    seen = set()
    unique = []
    duplicates = 0
    train_count = 0
    sources = []
    for index, path in enumerate(a.source):
        frames = read_xyz(path)
        sources.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "frames": len(frames),
            }
        )
        for frame in frames:
            key = identity(frame)
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            unique.append(frame)
        if index == 0:
            train_count = len(unique)
    rng = np.random.default_rng(a.seed)
    if a.preserve_training:
        tail = rng.permutation(np.arange(train_count, len(unique)))
        partitions = [
            np.arange(train_count),
            tail[: len(tail) // 2],
            tail[len(tail) // 2 :],
        ]
    else:
        idx = rng.permutation(len(unique))
        n = len(idx)
        partitions = [
            idx[: int(0.8 * n)],
            idx[int(0.8 * n) : int(0.9 * n)],
            idx[int(0.9 * n) :],
        ]
    if min(map(len, partitions)) < 1:
        raise ValueError("Not enough distinct frames for three nonempty partitions")
    a.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "seed": a.seed,
        "sources": sources,
        "duplicate_geometries_removed": duplicates,
        "identity": "species + cell/positions rounded to 9 decimals; no rotation/permutation equivalence",
        "split": "preserve first source as training"
        if a.preserve_training
        else "80/10/10 random unique geometries",
        "limitations": "Random structure split; trajectory/material-domain holdouts still required.",
        "partitions": {},
    }
    for name, indices in zip(("train", "validation", "test"), partitions):
        path = a.output / (name + ".xyz")
        path.write_text("".join(unique[i].text for i in indices))
        manifest["partitions"][name] = {
            "frames": len(indices),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "unique_source_indices": indices.tolist(),
        }
    (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {k: manifest[k] for k in ["seed", "duplicate_geometries_removed", "split"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

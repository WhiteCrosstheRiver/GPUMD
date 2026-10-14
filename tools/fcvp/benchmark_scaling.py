#!/usr/bin/env python3
"""Paired one/two-GPU wall-time comparison with identical data and initialization."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time

import numpy as np

from benchmark_gpu import initial_parameters
from reference import read_xyz


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gnep", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument(
        "--repeat-training",
        type=int,
        default=1,
        help="Repeat frames for workload scaling only, not new training data",
    )
    parser.add_argument("--seconds", type=float, default=1800)
    args = parser.parse_args()
    devices = args.devices.split(",")
    if len(devices) != 2 or len(set(devices)) != 2 or not all(devices):
        parser.error("--devices must identify two distinct GPUs")
    if (
        min(args.epochs, args.repeats, args.batch, args.interval, args.repeat_training)
        < 1
    ):
        parser.error("Counts must be positive")
    args.gnep = args.gnep.resolve()
    args.data = args.data.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    elements = sorted({s for f in read_xyz(args.data / "train.xyz") for s in f.species})
    initial, _ = initial_parameters(args.seed, len(elements), 30)
    config = (
        f"type {len(elements)} " + " ".join(elements) + "\n"
        "cutoff 5 4\nn_max 4 4\nbasis_size 8 8\nneuron 30\nl_max 4\n"
        f"batch {args.batch}\nepoch {args.epochs}\nseed {args.seed}\n"
        "lambda_e 1\nlambda_f 2\nlambda_v 0\nweight_decay 0\n"
        f"fcvp {args.interval} 40 1e-8 1e-8\n"
    )
    report = {
        "scope": "Fixed-work FCVP training scaling, not time to an accuracy threshold",
        "config": config,
        "training_repetitions": args.repeat_training,
        "repetition_note": "Repeated frames measure workload scaling only; they are not independent samples.",
        "binary_sha256": hashlib.sha256(args.gnep.read_bytes()).hexdigest(),
        "data_sha256": {
            name: hashlib.sha256((args.data / (name + ".xyz")).read_bytes()).hexdigest()
            for name in ("train", "validation")
        },
        "gpu": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version",
                "--format=csv",
            ],
            text=True,
        ),
        "runs": [],
    }
    for repeat in range(args.repeats):
        order = (1, 2) if repeat % 2 == 0 else (2, 1)
        for count in order:
            directory = args.output / f"repeat{repeat}_gpu{count}"
            directory.mkdir()
            (directory / "gnep.in").write_text(config)
            np.savetxt(directory / "gnep.restart", initial, fmt="%.9e")
            for name, source in (("train", "train"), ("test", "validation")):
                if name == "train" and args.repeat_training > 1:
                    (directory / "train.xyz").write_bytes(
                        (args.data / "train.xyz").read_bytes() * args.repeat_training
                    )
                else:
                    shutil.copyfile(
                        args.data / (source + ".xyz"), directory / (name + ".xyz")
                    )
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = ",".join(devices[:count])
            start = time.perf_counter()
            with (directory / "stdout.log").open("w") as log:
                subprocess.run(
                    [str(args.gnep)],
                    cwd=directory,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=args.seconds,
                )
            wall = time.perf_counter() - start
            losses = np.loadtxt(directory / "loss.out", ndmin=2)
            if len(losses) != args.epochs or not np.isfinite(losses).all():
                raise RuntimeError(f"Invalid/incomplete loss trace in {directory}")
            final = losses[-1]
            report["runs"].append(
                {
                    "repeat": repeat,
                    "gpus": count,
                    "wall_seconds": wall,
                    "training_seconds": float(final[-1]),
                    "validation_rmse": final[5:8].tolist(),
                    "model_sha256": hashlib.sha256(
                        (directory / "nep.txt").read_bytes()
                    ).hexdigest(),
                }
            )
            (args.output / "summary.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            print(repeat, count, wall, final[5:8], flush=True)
    report["median_wall_seconds"] = {
        str(count): statistics.median(
            r["wall_seconds"] for r in report["runs"] if r["gpus"] == count
        )
        for count in (1, 2)
    }
    report["median_training_seconds"] = {
        str(count): statistics.median(
            r["training_seconds"] for r in report["runs"] if r["gpus"] == count
        )
        for count in (1, 2)
    }
    report["wall_speedup"] = (
        report["median_wall_seconds"]["1"] / report["median_wall_seconds"]["2"]
    )
    report["training_speedup"] = (
        report["median_training_seconds"]["1"] / report["median_training_seconds"]["2"]
    )
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

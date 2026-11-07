#!/usr/bin/env python3
"""Independently check paired scaling models outside the timed GPU experiment."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmark_cpu import metrics
from reference import Oracle, read_xyz


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.report.read_text())
    frames = {
        name: read_xyz(args.data / (name + ".xyz")) for name in ("validation", "test")
    }
    report = {
        "scope": "CPU oracle checks of repeat-0 final models, not time to target",
        "models": [],
    }
    for run in summary["runs"]:
        if run["repeat"] != 0:
            continue
        model = args.report.parent / f"repeat0_gpu{run['gpus']}" / "nep.txt"
        digest = hashlib.sha256(model.read_bytes()).hexdigest()
        if digest != run["model_sha256"]:
            raise RuntimeError(f"Model changed since benchmark: {model}")
        oracle = Oracle(args.bridge, model)
        try:
            result = {
                name: metrics(oracle, items, oracle.model.beta)
                for name, items in frames.items()
            }
        finally:
            oracle.close()
        for measured, native in zip(
            result["validation"].values(), run["validation_rmse"]
        ):
            if measured is not None and not np.isclose(
                measured, native, atol=5e-5, rtol=2e-4
            ):
                raise RuntimeError(f"Native/CPU validation mismatch: {model}")
        report["models"].append({"gpus": run["gpus"], "sha256": digest, **result})
    if len(report["models"]) != 2:
        raise RuntimeError("Expected two repeat-0 models")
    for name in frames:
        left, right = (m[name] for m in report["models"])
        for key in left:
            if left[key] is not None:
                np.testing.assert_allclose(
                    left[key],
                    right[key],
                    atol=5e-5,
                    rtol=2e-3,
                    err_msg=f"Single/dual {name} {key}",
                )
    report["passed"] = True
    output = args.report.parent / "independent_quality.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

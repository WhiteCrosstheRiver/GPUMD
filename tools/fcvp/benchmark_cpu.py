#!/usr/bin/env python3
"""Real-XYZ, frozen-representation benchmark. NEVER a GPU/whole-training speedup."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
from reference import (
    Oracle,
    read_xyz,
    groups_from_basis,
    objective,
    reference_mm,
    NEP_CPU_COMMIT,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(oracle, frames, beta):
    e = []
    f = []
    v = []
    for frame in frames:
        pred = oracle.predict(frame, beta)
        n = len(frame.species)
        e.extend(pred[:1] - frame.energy)
        f.extend(pred[1 : 1 + 3 * n] - frame.force.T.ravel())
        if frame.virial is not None:
            v.extend(pred[-6:] - frame.virial)
    return dict(
        energy_eV_atom=float(np.sqrt(np.mean(np.square(e)))),
        force_eV_A=float(np.sqrt(np.mean(np.square(f)))),
        virial_eV_atom=float(np.sqrt(np.mean(np.square(v)))) if v else None,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bridge", "model", "train", "test", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--lambda1", type=float, default=0.0)
    parser.add_argument("--lambda2", type=float, default=0.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    train = read_xyz(args.train)
    test = read_xyz(args.test)

    # Reject exact geometry leakage even if comment formatting/labels differ.
    def identity(frame):
        return (
            tuple(frame.species),
            np.round(frame.cell, 9).tobytes(),
            np.round(frame.positions, 9).tobytes(),
        )

    overlap = set(map(identity, train)) & set(map(identity, test))
    if overlap:
        raise ValueError(f"Train/test overlap: {len(overlap)} geometries")
    oracle = Oracle(args.bridge, args.model)
    beta = oracle.model.beta
    provenance = (
        json.loads(args.bridge.with_suffix(".json").read_text())
        if args.bridge.with_suffix(".json").exists()
        else {}
    )
    if provenance and provenance["binary_sha256"] != sha(args.bridge):
        raise ValueError("Bridge binary differs from build manifest")
    before = metrics(oracle, test, beta)
    t = time.perf_counter()
    bases = [oracle.basis(frame) for frame in train]
    basis_seconds = time.perf_counter() - t
    errors = np.zeros(3)
    for frame, (a, offset) in zip(train, bases):
        delta = np.abs(a @ beta + offset - oracle.predict(frame, beta))
        errors = np.maximum(
            errors, [np.max(delta[:1]), np.max(delta[1:-6]), np.max(delta[-6:])]
        )
    groups = groups_from_basis(train, bases)
    opts = dict(iterations=args.iterations, lambda1=args.lambda1, lambda2=args.lambda2)
    t = time.perf_counter()
    fitted, history = oracle.solve(groups, beta, **opts)
    solve_seconds = time.perf_counter() - t
    fixed = oracle.model.fixed_regularization_parameters()
    t = time.perf_counter()
    ref, refhistory = reference_mm(
        groups, beta, **opts, fixed=fixed, total=oracle.model.training_parameter_count
    )
    reference_seconds = time.perf_counter() - t
    stacked = np.vstack([a / np.sqrt(len(y)) for a, y, w in groups if w])
    singular = np.linalg.svd(stacked, compute_uv=False)
    oracle.model.write(args.output / "nep_fcvp.txt", fitted)
    after = metrics(oracle, test, fitted)
    report = {
        "scope": "CPU frozen-output refit on real XYZ, not end-to-end training or GPU speedup",
        "source_model": "Supplied pretrained representation remains fixed; only output weights/biases change",
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "gpu_available": False,
            "nep_cpu_commit": provenance.get("nep_cpu_commit"),
            "bridge_build": provenance,
        },
        "inputs": {
            k: {"path": str(getattr(args, k)), "sha256": sha(getattr(args, k))}
            for k in ("model", "train", "test")
        },
        "configurations": {
            "train": len(train),
            "test": len(test),
            "train_test_geometry_overlap": len(overlap),
            "train_with_virial": sum(f.virial is not None for f in train),
        },
        "parameters": {
            "total": oracle.model.training_parameter_count,
            "linear": len(beta),
        },
        "linearity_max_abs_E_F_V": errors.tolist(),
        "design_matrix_rank": int(np.sum(singular > singular[0] * 1e-12)),
        "design_matrix_condition": float(singular[0] / singular[-1]),
        "gram_condition_estimate": float((singular[0] / singular[-1]) ** 2),
        "mm_objectives": history.tolist(),
        "mm_steps": len(history) - 1,
        "mm_monotone": bool(np.all(np.diff(history) <= 0)),
        "independent_svd_final_objective": float(refhistory[-1]),
        "objective_gap_to_reference": float(history[-1] - refhistory[-1]),
        "timing_seconds": {
            "basis": basis_seconds,
            "gram_and_mm": solve_seconds,
            "svd_reference_mm": reference_seconds,
            "projection_total": basis_seconds + solve_seconds,
        },
        "test_before": before,
        "test_after": after,
        "gpu_training_speedup": None,
        "snes_time_to_target": None,
        "gnep_time_to_target": None,
        "fcvp_time_to_target": None,
        "limitations": [
            "Frozen representation only; timings do not compare full optimizers.",
            "No GPU execution; no claim of 1.5x or larger training acceleration.",
            "Missing virial labels are excluded, never replaced by zero targets.",
            "No SiGe transfer, EOS, or MD stability acceptance has been performed.",
        ],
    }
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "configurations",
                    "linearity_max_abs_E_F_V",
                    "mm_steps",
                    "mm_monotone",
                    "objective_gap_to_reference",
                    "timing_seconds",
                    "test_before",
                    "test_after",
                )
            },
            indent=2,
        )
    )
    oracle.close()


if __name__ == "__main__":
    main()

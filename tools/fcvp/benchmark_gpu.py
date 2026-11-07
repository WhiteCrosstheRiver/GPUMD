#!/usr/bin/env python3
"""Run timed GPU training baselines and FCVP; targets use validation data only.

The runner timestamps observed validation reports from process launch, including
startup, projection, and validation overhead. It never infers speed from epochs.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import subprocess
import threading
import time
import numpy as np
from reference import Oracle, read_xyz
from benchmark_cpu import metrics
from prepare_data import identity


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_commit(binary):
    result = subprocess.run(
        ["git", "-C", str(binary.parent), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def meets_targets(values, targets):
    return all(
        limit is None or (value is not None and np.isfinite(value) and value <= limit)
        for value, limit in zip(values, targets)
    )


def run(binary, directory, environment, targets, timeout, snes=False):
    start = time.perf_counter()
    stop = threading.Event()
    expired = threading.Event()
    peak = []
    trace = []
    proc = subprocess.Popen(
        [str(binary)],
        cwd=directory,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def sample_memory():
        while not stop.is_set():
            try:
                text = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=pid,used_gpu_memory",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                )
                memory = [
                    int(row.split(",")[1])
                    for row in text.splitlines()
                    if int(row.split(",")[0]) == proc.pid
                ]
                if memory:
                    peak.append(sum(memory))
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            stop.wait(0.25)

    def expire():
        expired.set()
        if proc.poll() is None:
            proc.kill()

    monitor = threading.Thread(target=sample_memory, daemon=True)
    monitor.start()
    watchdog = threading.Timer(timeout, expire)
    watchdog.start()
    reached = None
    try:
        with (directory / "stdout.log").open("w") as log:
            for line in proc.stdout:
                log.write(line)
                log.flush()
                fields = line.split()
                if len(fields) != 10:
                    continue
                try:
                    values = list(map(float, fields))
                except ValueError:
                    continue
                if not fields[0].isdigit():
                    continue
                timestamp = time.perf_counter() - start
                validation = values[7:10] if snes else values[5:8]
                trace.append(
                    {
                        "iteration": int(values[0]),
                        "seconds": timestamp,
                        "validation_rmse": validation,
                    }
                )
                if meets_targets(validation, targets):
                    reached = trace[-1]
                    # Freeze the process while copying; verify this snapshot against
                    # the reported validation result with the independent oracle below.
                    if os.name == "posix" and proc.poll() is None:
                        proc.send_signal(signal.SIGSTOP)
                        os.waitpid(proc.pid, os.WUNTRACED)
                        try:
                            shutil.copyfile(
                                directory / "nep.txt", directory / "nep_at_target.txt"
                            )
                        finally:
                            proc.send_signal(signal.SIGCONT)
                    else:
                        shutil.copyfile(
                            directory / "nep.txt", directory / "nep_at_target.txt"
                        )
                    if proc.poll() is None:
                        proc.terminate()
                    break
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    finally:
        watchdog.cancel()
        stop.set()
        monitor.join(timeout=6)
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        proc.stdout.close()
    elapsed = time.perf_counter() - start
    (directory / "validation_trace.json").write_text(json.dumps(trace, indent=2) + "\n")
    return {
        "time_to_target_seconds": reached["seconds"] if reached else None,
        "target_checkpoint": reached,
        "process_wall_seconds": elapsed,
        "peak_gpu_memory_sampled_MiB": max(peak) if peak else None,
        "peak_memory_note": "250 ms process-memory sampling, not an exact allocator high-water mark",
        "exit_code": proc.returncode,
        "timed_out": expired.is_set(),
        "status": "target_reached"
        if reached
        else (
            "budget_exhausted" if proc.returncode == 0 or expired.is_set() else "failed"
        ),
    }


def initial_parameters(seed, types, neurons):
    rng = np.random.default_rng(seed)
    dim = 25
    blocks = []
    for _ in range(types):
        blocks.append(
            np.r_[
                rng.normal(0, np.sqrt(2 / (dim + neurons)), dim * neurons),
                rng.normal(0, 0.01, neurons),
                rng.normal(0, np.sqrt(2 / (neurons + 1)), neurons),
                0.0,
            ]
        )
    coefficients = 0.1 * rng.normal(size=types * types * 90) + rng.uniform(
        -0.5, 0.5, size=types * types * 90
    )
    return np.r_[np.concatenate(blocks), coefficients], np.r_[
        np.concatenate([b[:-1] for b in blocks]), 0.0, coefficients
    ]


def gnep_config(common, epochs, seed, extra, upstream=False):
    config = common + f"l_max 4\nepoch {epochs}\nweight_decay 0\n"
    # The upstream parser predates `seed`; the saved restart still fixes its
    # initial parameters, but upstream batch shuffling remains nondeterministic.
    if not upstream:
        config += f"seed {seed}\n"
    return config + extra


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("gnep", "bridge", "data", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--upstream-gnep", type=Path)
    p.add_argument("--snes", type=Path)
    p.add_argument("--device", default="0")
    p.add_argument(
        "--build-notes",
        default="unspecified; supply compiler/toolkit/architecture flags",
    )
    p.add_argument(
        "--seeds", type=int, nargs="+", default=[20261014, 20261015, 20261016]
    )
    p.add_argument("--intervals", type=int, nargs="+", default=[1, 5, 10])
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--generations", type=int, default=200000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--neurons", type=int, default=30)
    p.add_argument("--seconds", type=float, default=3600)
    p.add_argument(
        "--energy-target",
        type=float,
        required=True,
        help="eV/atom, fixed before running",
    )
    p.add_argument(
        "--force-target",
        type=float,
        required=True,
        help="eV/Angstrom, fixed before running",
    )
    p.add_argument(
        "--virial-target",
        type=float,
        default=None,
        help="eV/atom, requires labeled validation virials",
    )
    a = p.parse_args()
    devices = a.device.split(",")
    if (
        not all(devices)
        or len(set(devices)) != len(devices)
        or min(a.intervals) < 1
        or min(a.epochs, a.generations, a.batch, a.neurons) < 1
    ):
        p.error("Use distinct visible GPUs and positive intervals/counts")
    if any(
        not np.isfinite(x) or x <= 0
        for x in (a.energy_target, a.force_target, a.seconds)
    ):
        p.error("Targets/time budget must be finite and positive")
    if a.virial_target is not None and (
        not np.isfinite(a.virial_target) or a.virial_target <= 0
    ):
        p.error("Virial target must be finite and positive")
    if not shutil.which("nvidia-smi"):
        p.error("NVIDIA GPU runtime is required; CPU timings are not a substitute")
    for name in ("gnep", "upstream_gnep", "snes", "bridge"):
        if getattr(a, name):
            setattr(a, name, getattr(a, name).resolve())
    a.data = a.data.resolve()
    a.output = a.output.resolve()
    frames = {
        name: read_xyz(a.data / (name + ".xyz"))
        for name in ("train", "validation", "test")
    }
    sets = {k: set(map(identity, v)) for k, v in frames.items()}
    if (
        sets["train"] & sets["validation"]
        or sets["train"] & sets["test"]
        or sets["validation"] & sets["test"]
    ):
        p.error("Data partitions overlap")
    if a.virial_target is not None and not any(
        f.virial is not None for f in frames["validation"]
    ):
        p.error("Requested virial threshold but no validation virial labels")
    elements = sorted({t for fs in frames.values() for f in fs for t in f.species})
    types = len(elements)
    has_virial = any(f.virial is not None for f in frames["train"])
    if a.virial_target is not None and not has_virial:
        p.error("Virial training labels required")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = a.device
    a.output.mkdir(parents=True, exist_ok=False)
    gpu_info = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total",
            "--format=csv",
        ],
        text=True,
    )
    common = (
        f"type {types} "
        + " ".join(elements)
        + f"\ncutoff 5 4\nn_max 4 4\nbasis_size 8 8\nneuron {a.neurons}\n"
    )
    common += f"batch {a.batch}\nlambda_e 1\nlambda_f 2\nlambda_v {0.1 if has_virial else 0}\n"
    report = {
        "scope": "GPU training time to preregistered validation thresholds; test held out",
        "targets": [a.energy_target, a.force_target, a.virial_target],
        "gpu": gpu_info,
        "binary_sha256": {
            k: digest(getattr(a, k))
            for k in ("gnep", "upstream_gnep", "snes")
            if getattr(a, k)
        },
        "data_sha256": {k: digest(a.data / (k + ".xyz")) for k in frames},
        "initialization": "Identical saved NEP5 initial vectors for all Adam/FCVP variants per seed",
        "upstream_gnep_caveat": "The unpatched upstream parser has no seed keyword. Its saved restart matches the other NEP5 variants, but its original batch shuffling remains nondeterministic.",
        "snes_caveat": "This branch trains NEP4 only; its shared bias differs from NEP5 per-type biases. SNES is context, not the primary controlled speedup comparison.",
        "build_notes": a.build_notes,
        "python": sys.version,
        "numpy": np.__version__,
        "source_commits": {
            k: source_commit(getattr(a, k))
            for k in ("gnep", "upstream_gnep", "snes")
            if getattr(a, k)
        },
        "runs": [],
    }
    report_path = a.output / "summary.json"
    # A separate correctness run prevents failed gradients from producing a speed claim.
    check = a.output / "gpu_correctness_gate"
    check.mkdir()
    for name, source in [("train", "train"), ("test", "validation")]:
        shutil.copyfile(a.data / (source + ".xyz"), check / (name + ".xyz"))
    (check / "gnep.in").write_text(
        common + "l_max 4\nepoch 1\nseed 1234567\nfcvp 1 10 1e-8 1e-8\nfcvp_check 1\n"
    )
    with (check / "stdout.log").open("w") as log:
        subprocess.run(
            [str(a.gnep)],
            cwd=check,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=a.seconds,
        )
    for seed in a.seeds:
        nep5, nep4 = initial_parameters(seed, types, a.neurons)
        variants = [
            ("adam_legacy", a.gnep, ""),
            ("adam_rmse", a.gnep, "loss_rmse 1e-8\n"),
        ]
        variants += [
            (f"fcvp_{k}", a.gnep, f"fcvp {k} 40 1e-8 1e-8\n") for k in a.intervals
        ]
        if a.upstream_gnep:
            variants.append(("upstream_gnep", a.upstream_gnep, ""))
        if a.snes:
            variants.append(("snes", a.snes, ""))
        for label, binary, extra in variants:
            directory = a.output / f"{label}_seed{seed}"
            directory.mkdir()
            is_snes = label == "snes"
            for name, source in [("train", "train"), ("test", "validation")]:
                shutil.copyfile(a.data / (source + ".xyz"), directory / (name + ".xyz"))
            if is_snes:
                config = (
                    common
                    + f"l_max 4 0 0\nversion 4\ngeneration {a.generations}\npopulation 50\nlambda_1 0\nlambda_2 0\n"
                )
                (directory / "nep.in").write_text(config)
                np.savetxt(
                    directory / "nep.restart",
                    np.column_stack([nep4, np.full(len(nep4), 0.1)]),
                    fmt="%.9e",
                )
            else:
                config = gnep_config(
                    common, a.epochs, seed, extra, label == "upstream_gnep"
                )
                (directory / "gnep.in").write_text(config)
                np.savetxt(directory / "gnep.restart", nep5, fmt="%.9e")
            result = run(binary, directory, env, report["targets"], a.seconds, is_snes)
            result.update(method=label, seed=seed, input=config)
            checkpoint = directory / "nep_at_target.txt"
            if checkpoint.exists():
                oracle = Oracle(a.bridge, checkpoint)
                try:
                    validation = metrics(
                        oracle, frames["validation"], oracle.model.beta
                    )
                    result["validation_checkpoint_rmse"] = validation
                    measured = list(validation.values())
                    reported = result["target_checkpoint"]["validation_rmse"]
                    # Allow printed precision and CPU/GPU roundoff, but never
                    # accept a checkpoint which actually misses a target.
                    matches = all(
                        v is None or np.isclose(v, r, atol=5e-5, rtol=2e-4)
                        for v, r in zip(measured, reported)
                    )
                    if not matches or not meets_targets(measured, report["targets"]):
                        result["status"] = "checkpoint_validation_failed"
                        result["time_to_target_seconds"] = None
                    else:
                        result["heldout_test_rmse"] = metrics(
                            oracle, frames["test"], oracle.model.beta
                        )
                        result["heldout_targets_met"] = meets_targets(
                            list(result["heldout_test_rmse"].values()),
                            report["targets"],
                        )
                finally:
                    oracle.close()
            report["runs"].append(result)
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            print(
                label,
                seed,
                result["status"],
                result["time_to_target_seconds"],
                flush=True,
            )
    # Missing/failed runs remain null and never count as a speedup.
    for control in ("adam_rmse", "adam_legacy", "upstream_gnep", "snes"):
        baseline_runs = {r["seed"]: r for r in report["runs"] if r["method"] == control}
        if not baseline_runs:
            continue
        paired = []
        for r in report["runs"]:
            if r["method"].startswith("fcvp_"):
                baseline = baseline_runs[r["seed"]]
                t0 = baseline["time_to_target_seconds"]
                t1 = r["time_to_target_seconds"]
                quality = baseline.get("heldout_targets_met", False) and r.get(
                    "heldout_targets_met", False
                )
                paired.append(
                    {
                        "seed": r["seed"],
                        "method": r["method"],
                        "validation_speedup": t0 / t1 if t0 and t1 else None,
                        "both_heldout_targets_met": quality,
                        "quality_gated_speedup": t0 / t1
                        if t0 and t1 and quality
                        else None,
                    }
                )
        report["paired_speedups_vs_" + control] = paired
    report_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

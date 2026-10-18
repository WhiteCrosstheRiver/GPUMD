"""Bounded, free-GPU-only acceptance for the isolated SOAP GNEP experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np


CELL = 'Lattice="40 0 0 0 40 0 0 0 40" '
PROPERTIES = "Properties=species:S:1:pos:R:3:force:R:3\n"
VIRIAL = 'virial="1 .2 -.3 .2 2 .4 -.3 .4 3" '
FRAMES = (
    "4\n" + CELL + "energy=-2 weight=1.3 " + VIRIAL + PROPERTIES
    + "Si 2 2 2 .1 .2 -.1\nGe 3.7 2.2 2.3 -.2 .1 .3\n"
    + "Si 2.5 3.8 2.4 .2 -.1 .1\nGe 3.1 2.7 4.1 -.1 -.2 -.3\n",
    "3\n" + CELL + "energy=-1 weight=.7 " + VIRIAL + PROPERTIES
    + "Ge 8 8 8 .1 -.1 .2\nSi 9.5 8.7 8.3 -.1 .1 -.2\n"
    + "Ge 8.4 9.6 8.9 .2 .1 -.1\n",
    "2\n" + CELL + "energy=.2 weight=.9 " + VIRIAL + PROPERTIES
    + "Si 15 15 15 0 0 0\nGe 17 16 15.5 .1 -.1 .2\n",
    "4\n" + CELL + "energy=-1.4 weight=1.1 " + VIRIAL + PROPERTIES
    + "Ge 20 20 20 -.1 .1 -.2\nSi 21.7 20.2 20.3 .2 -.1 .2\n"
    + "Ge 20.5 21.8 20.4 -.2 .2 .1\nSi 21.1 20.7 22.1 .1 -.2 -.1\n",
)
BASE = (
    "type 2 Si Ge\ncutoff 5 4\nn_max 2 2\nbasis_size 3 3\n"
    "l_max 4 2 1\nneuron 8\nbatch 2 0\nepoch 3\n"
    "weight_decay 0\nseed 184\nstart_lr .001\nstop_lr .001\n"
    "lambda_e 1\nlambda_f 2\nlambda_v .1\n"
    "lambda_shear 1.5\ntype_weight 1.2 .8\nforce_delta .2\n"
    "loss_rmse 1e-3\n"
)
PARAMETER_COUNT = ((21 + 2) * 8 + 1) * 2 + 2 * 2 * (3 * 4 + 3 * 4)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def free_gpu(gpu):
    inventory = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        text=True, timeout=30,
    )
    lookup = {line.split(",")[0].strip(): line.split(",")[1].strip()
              for line in inventory.splitlines() if line.strip()}
    if gpu not in lookup:
        raise RuntimeError(f"GPU index {gpu} not in nvidia-smi inventory")
    applications = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name",
         "--format=csv,noheader,nounits"], text=True, timeout=30,
    )
    busy = [line for line in applications.splitlines()
            if line.split(",")[0].strip() == lookup[gpu]]
    if busy:
        raise RuntimeError(f"Refusing to run on occupied GPU {gpu}: {busy}")
    return {"index": gpu, "uuid": lookup[gpu], "compute_processes": busy}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", required=True, help="One physical nvidia-smi GPU index")
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--bridge", required=True, type=Path)
    parser.add_argument("--tools", required=True, type=Path, help="Directory containing FCVP reference.py")
    parser.add_argument("--output", required=True, type=Path, help="New acceptance directory")
    args = parser.parse_args()
    if not args.gpu.isdigit():
        parser.error("--gpu must be one physical numeric GPU index")
    for field in ("binary", "baseline", "bridge", "tools", "output"):
        setattr(args, field, getattr(args, field).resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.tools))
    reference = importlib.import_module("reference")
    initial = np.random.default_rng(20261016).integers(-100, 101, PARAMETER_COUNT) / 1024.0
    restart = "".join(f"{value:.9e}\n" for value in initial)
    result = {
        "passed": False,
        "binary": str(args.binary), "binary_sha256": sha256(args.binary),
        "baseline": str(args.baseline), "baseline_sha256": sha256(args.baseline),
        "bridge": str(args.bridge), "bridge_sha256": sha256(args.bridge),
        "checks": [], "runs": {},
    }
    begin = time.monotonic()

    def run(name, config, mode=None, binary=None, model=None, expect_success=True):
        guard = free_gpu(args.gpu)
        directory = args.output / name
        directory.mkdir()
        for filename in ("train.xyz", "test.xyz"):
            (directory / filename).write_text("".join(FRAMES))
        (directory / "gnep.in").write_text(config)
        if model is None:
            (directory / "gnep.restart").write_text(restart)
        else:
            (directory / "nep.txt").write_bytes(Path(model).read_bytes())
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = args.gpu
        environment.pop("GPUMD_EXPERIMENT_OPTIMIZER", None)
        if mode is not None:
            environment["GPUMD_EXPERIMENT_OPTIMIZER"] = mode
        start = time.monotonic()
        process = subprocess.run(
            [str(binary or args.binary)], cwd=directory, env=environment,
            capture_output=True, text=True, timeout=240,
        )
        log = process.stdout + process.stderr
        (directory / "stdout.log").write_text(log)
        result["runs"][name] = {
            "returncode": process.returncode, "elapsed_seconds": time.monotonic() - start,
            "mode": mode, "guard": guard,
        }
        if expect_success and process.returncode != 0:
            raise AssertionError(f"{name} failed with {process.returncode}: {log[-8000:]}")
        if not expect_success and process.returncode == 0:
            raise AssertionError(f"{name} incorrectly accepted invalid experimental configuration")
        return directory, log

    def check(name, details=None):
        result["checks"].append({"name": name, "passed": True, **(details or {})})

    def native_cpu(directory, name):
        model_path = directory / "nep.txt"
        model = reference.Model.read(model_path)
        frames = reference.read_xyz(directory / "test.xyz")
        oracle = reference.Oracle(args.bridge, model_path)
        try:
            expected = [oracle.predict(frame, model.beta) for frame in frames]
        finally:
            oracle.close()
        predicted, _ = run(name, "prediction 1\nbatch 2\n", model=model_path)
        actual = {
            quantity: np.loadtxt(predicted / f"{quantity}_train.out", ndmin=2)
            for quantity in ("energy", "force", "virial")
        }
        wanted = {
            "energy": np.asarray([row[0] for row in expected]),
            "force": np.concatenate([
                row[1:1 + 3 * len(frame.species)].reshape(3, -1).T
                for frame, row in zip(frames, expected)
            ]),
            "virial": np.asarray([row[-6:] for row in expected]),
        }
        observed = {
            "energy": actual["energy"][:, 0],
            "force": actual["force"][:, :3],
            "virial": actual["virial"][:, :6],
        }
        differences = {}
        for quantity in wanted:
            np.testing.assert_allclose(observed[quantity], wanted[quantity], rtol=2e-3, atol=2e-3)
            differences[quantity + "_max_abs"] = float(np.max(np.abs(observed[quantity] - wanted[quantity])))
        check(name + "_cpu_efv", differences)

    try:
        result["initial_gpu"] = free_gpu(args.gpu)
        baseline, _ = run("baseline_adam", BASE, binary=args.baseline)
        default, _ = run("new_default", BASE)
        explicit, _ = run("new_explicit_adam", BASE, "adam")
        for name, directory in (("default_path", default), ("explicit_adam", explicit)):
            for filename in ("nep.txt", "gnep.restart"):
                if (directory / filename).read_bytes() != (baseline / filename).read_bytes():
                    raise AssertionError(f"{name} changed {filename} versus native Adam")
            left = np.loadtxt(directory / "loss.out", ndmin=2)
            right = np.loadtxt(baseline / "loss.out", ndmin=2)
            np.testing.assert_array_equal(left[:, :-1], right[:, :-1])
            check(name + "_exact_model_restart_loss")

        for mode in ("soap", "adam_soap_control"):
            directory, log = run(mode, BASE + "fcvp_check 1\n", mode)
            if "analytical gradient gate passed" not in log:
                raise AssertionError(f"{mode} did not pass the analytical gradient gate")
            counters = re.findall(r"calls=(\d+) updates=(\d+) host_seconds=([\d.eE+-]+) transfer_seconds=([\d.eE+-]+)", log)
            if not counters or tuple(map(int, counters[-1][:2])) != (6, 5):
                raise AssertionError(f"{mode} did not produce six calls/five updates: {counters}")
            loss = np.loadtxt(directory / "loss.out", ndmin=2)
            values = np.loadtxt(directory / "gnep.restart")
            if loss.shape != (3, 10) or not np.isfinite(loss).all() or not np.isfinite(values).all():
                raise AssertionError(f"{mode} produced invalid loss or parameters")
            json.loads((directory / "experimental_soap_state.json").read_text())
            check(mode + "_finite_counters_gradient_gate", {
                "calls": 6, "updates": 5, "last_loss": loss[-1].tolist(),
                "host_seconds": float(counters[-1][2]), "transfer_seconds": float(counters[-1][3]),
            })
            native_cpu(directory, mode + "_predict")
            first, log = run(
                mode + "_first_skip",
                BASE.replace("batch 2 0", "batch 1000 0").replace("epoch 3", "epoch 1"), mode,
            )
            first_state = json.loads((first / "experimental_soap_state.json").read_text())
            np.testing.assert_array_equal(
                np.asarray(first_state["params"]), initial.astype(np.float32).astype(np.float64))
            # Native checkpoints print eight significant digits, not a float32 round trip.
            expected_restart = "".join(f"{float(value):15.7e}\n" for value in initial.astype(np.float32))
            if (first / "gnep.restart").read_text() != expected_restart:
                raise AssertionError(f"{mode} checkpoint differs from unchanged parameters at native precision")
            if first_state["calls"] != 1 or first_state["updates"] != 0:
                raise AssertionError(f"{mode} diagnostic state reports an unexpected first update")
            if "calls=1 updates=0" not in log:
                raise AssertionError(f"{mode} first call did not skip the update")
            check(mode + "_first_step_skip")

        zbl, _ = run("soap_zbl", BASE + "zbl 2.4\n", "soap")
        if not reference.Model.read(zbl / "nep.txt").header[0].startswith("nep5_zbl "):
            raise AssertionError("SOAP model did not preserve embedded ZBL")
        native_cpu(zbl, "soap_zbl_predict")
        for name, config, mode, message in (
            ("reject_unknown", BASE, "not_an_optimizer", "must be adam, soap, or adam_soap_control"),
            ("reject_no_rmse", BASE.replace("loss_rmse 1e-3\n", ""), "soap", "requires loss_rmse"),
            ("reject_fcvp", BASE + "fcvp 1 10 1e-3 1e-8\n", "soap", "does not combine with FCVP"),
        ):
            _, log = run(name, config, mode, expect_success=False)
            if message not in log:
                raise AssertionError(f"{name} failed for an unexpected reason: {log[-4000:]}")
            check(name)
        result["passed"] = True
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - begin
        (args.output / "verification.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

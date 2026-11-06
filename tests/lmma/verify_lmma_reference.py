# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent NumPy verification of paper Algorithm 1, arXiv:1705.06693.

This deliberately uses the paper's squared-path sigma update. The author's
public Python implementation uses a different norm/chi sigma update.
"""

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np


def verify(executable, n, population, generations):
    result = subprocess.run(
        [str(executable), "--trace", str(n), str(population), str(generations)],
        text=True,
        capture_output=True,
        check=True,
    )
    states = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(states) == generations + 1
    mu = population // 2
    memory_count = 4 + int(math.floor(3 * math.log(n)))
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= weights.sum()
    mu_eff = 1 / (weights @ weights)
    c_sigma = 2 * population / n
    c_d = 1 / (1.5 ** np.arange(memory_count) * n)
    c_c = population / (4.0 ** np.arange(memory_count) * n)
    mean = (0.2 * np.sin(np.arange(1, n + 1) * 0.3)).astype(np.float32).astype(float)
    path = np.zeros(n)
    memory = np.zeros((memory_count, n))
    sigma = 0.1
    maximum_error = {}

    def compare(state, iteration):
        expected = {
            "dimension": n,
            "population": population,
            "parents": mu,
            "memory_count": memory_count,
            "iteration": iteration,
            "sigma": sigma,
            "mu_eff": mu_eff,
            "c_sigma": c_sigma,
            "weights": weights,
            "c_d": c_d,
            "c_c": c_c,
            "mean": mean,
            "path": path,
            "memory": memory.ravel(),
        }
        for name, wanted in expected.items():
            actual = np.asarray(state[name])
            wanted = np.asarray(wanted)
            error = float(np.max(np.abs(actual - wanted)))
            maximum_error[name] = max(maximum_error.get(name, 0.0), error)
            np.testing.assert_allclose(actual, wanted, rtol=2e-10, atol=2e-12,
                                       err_msg=f"iteration={iteration}, field={name}")

    compare(states[0], 0)
    offset = np.arange(1, population * n + 1)
    for generation in range(generations):
        z = (np.sin((generation + 1) * 0.7 + offset * 0.11)
             + np.cos((generation + 1) * 0.13 + offset * 0.017)).reshape(population, n)
        directions = z.copy()
        for j in range(min(generation, memory_count)):
            directions = ((1 - c_d[j]) * directions
                          + c_d[j] * np.outer(directions @ memory[j], memory[j]))
        candidates = (mean + sigma * directions).astype(np.float32).astype(float)
        objective = np.sum(candidates * candidates, axis=1)
        selected = np.argsort(objective, kind="stable")[:mu]
        weighted_z = weights @ z[selected]
        weighted_d = weights @ directions[selected]
        mean += sigma * weighted_d
        path = ((1 - c_sigma) * path
                + math.sqrt(mu_eff * c_sigma * (2 - c_sigma)) * weighted_z)
        memory = ((1 - c_c[:, None]) * memory
                  + np.sqrt(mu_eff * c_c * (2 - c_c))[:, None] * weighted_z)
        sigma *= math.exp(c_sigma / 2 * (path @ path / n - 1))
        compare(states[generation + 1], generation + 1)
    return {"dimension": n, "population": population, "generations": generations,
            "memory_count": memory_count, "status": "PASS",
            "maximum_absolute_error": maximum_error}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", type=Path, default=Path(__file__).with_name("test_lmma_core.exe"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cases = [(128, 10, 100), (257, 50, 100), (4025, 50, 35)]
    report = {"reference": "arXiv:1705.06693v1 Algorithm 1",
              "sigma_variant": "exp(c_sigma/2 * (path_squared/dimension - 1))",
              "tests": [verify(args.exe, *case) for case in cases]}
    encoded = json.dumps(report, indent=2)
    print(encoded)
    if args.output:
        args.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

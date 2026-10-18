#!/usr/bin/env python3
"""Independent CPU recurrence checks for the experimental C++ SOAP port."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Config:
    lr: float = 0.003
    beta1: float = 0.95
    beta2: float = 0.95
    shampoo_beta: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.0
    frequency: int = 10


class Block:
    def __init__(self, name, rows, cols, indices, precondition, config):
        self.name = name
        self.shape = (rows, cols)
        self.indices = np.asarray(indices, dtype=int)
        self.precondition = precondition
        self.config = config
        self.m = np.zeros(self.shape)
        self.v = np.zeros(self.shape)
        self.left = np.zeros((rows, rows))
        self.right = np.zeros((cols, cols))
        self.ql = np.eye(rows)
        self.qr = np.eye(cols)
        self.initialized = False
        self.updates = 0

    def project(self, x):
        return self.ql.T @ x @ self.qr if self.precondition else x.copy()

    def back(self, x):
        return self.ql @ x @ self.qr.T if self.precondition else x.copy()

    def covariance(self, g):
        b = self.config.shampoo_beta
        self.left = b * self.left + (1 - b) * (g @ g.T)
        self.right = b * self.right + (1 - b) * (g.T @ g)

    def update(self, params, gradient):
        g = gradient[self.indices].reshape(self.shape)
        if not self.initialized:
            if self.precondition:
                self.covariance(g)
                self.ql = np.linalg.eigh(self.left)[1][:, ::-1]
                self.qr = np.linalg.eigh(self.right)[1][:, ::-1]
            self.initialized = True
            return
        c = self.config
        self.updates += 1
        gp = self.project(g)
        self.m = c.beta1 * self.m + (1 - c.beta1) * gp
        self.v = c.beta2 * self.v + (1 - c.beta2) * gp**2
        alpha = c.lr * np.sqrt(1 - c.beta2**self.updates) / (1 - c.beta1**self.updates)
        direction = self.back(self.m / (np.sqrt(self.v) + c.eps))
        params[self.indices] -= alpha * direction.ravel()
        if c.weight_decay:
            params[self.indices] *= 1 - c.lr * c.weight_decay
        if self.precondition:
            m_physical = self.back(self.m)
            self.covariance(g)
            if self.updates % c.frequency == 0:
                il = np.argsort(-np.diag(self.ql.T @ self.left @ self.ql), kind="stable")
                ir = np.argsort(-np.diag(self.qr.T @ self.right @ self.qr), kind="stable")
                self.v = self.v[np.ix_(il, ir)]
                self.ql = np.linalg.qr(self.left @ self.ql[:, il])[0]
                self.qr = np.linalg.qr(self.right @ self.qr[:, ir])[0]
            self.m = self.project(m_physical)


def input_text(weights, gradients, blocks, config):
    lines = [f"{weights.size} {len(blocks)} {len(gradients)}"]
    lines.append(" ".join(str(x) for x in (
        config.lr, config.beta1, config.beta2, config.shampoo_beta,
        config.eps, config.weight_decay, config.frequency,
    )))
    for block in blocks:
        rows, cols = block.shape
        lines.append(f"{block.name} {rows} {cols} {int(block.precondition)}")
        lines.append(" ".join(map(str, block.indices)))
    lines.append(" ".join(format(x, ".17g") for x in weights))
    lines.extend(" ".join(format(x, ".17g") for x in g) for g in gradients)
    return "\n".join(lines) + "\n"


def matrix(group, key, shape):
    return np.asarray(group[key], dtype=float).reshape(shape)


def deterministic_case(rectangular=False, decay=0.0, frequency=10):
    rng = np.random.default_rng(20261016 + int(rectangular))
    specifications = [("ann", 4, 4, True), ("descriptor", 7, 7, True),
                      ("bias", 1, 10, False), ("scalar", 1, 1, False)]
    if rectangular:
        specifications = [("ann", 40, 42, True), ("descriptor", 7, 9, True),
                          ("bias", 1, 40, False), ("scalar", 1, 1, False)]
    n = sum(rows * cols for _, rows, cols, _ in specifications)
    order = rng.permutation(n)
    config = Config(weight_decay=decay, frequency=frequency)
    blocks = []
    start = 0
    for name, rows, cols, precondition in specifications:
        count = rows * cols
        blocks.append(Block(name, rows, cols, order[start:start + count], precondition, config))
        start += count
    weights = rng.normal(0, 0.2, n)
    gradients = rng.normal(size=(51, n))
    for block in blocks:
        if block.precondition:
            rows, cols = block.shape
            u = np.linalg.qr(rng.normal(size=(rows, rows)))[0]
            v = np.linalg.qr(rng.normal(size=(cols, cols)))[0]
            diagonal = np.zeros((rows, cols))
            np.fill_diagonal(diagonal, np.linspace(0.5, 3.0, min(rows, cols)))
            gradients[0, block.indices] = (u @ diagonal @ v.T).ravel()
    return weights, gradients, blocks, config


def verify_trace(executable, rectangular=False, decay=0.0, frequency=10):
    initial, gradients, blocks, config = deterministic_case(rectangular, decay, frequency)
    run = subprocess.run(
        [str(executable), "--trace"], input=input_text(initial, gradients, blocks, config),
        text=True, capture_output=True, check=True, timeout=180,
    )
    records = [json.loads(line) for line in run.stdout.splitlines() if line.strip()]
    if len(records) != len(gradients) + 1:
        raise AssertionError(f"Expected {len(gradients) + 1} records, got {len(records)}")
    np.testing.assert_array_equal(records[0]["params"], initial)
    params = initial.copy()
    parameter_error = moment_error = covariance_error = orthogonality_error = 0.0
    for call, gradient in enumerate(gradients, start=1):
        for block in blocks:
            block.update(params, gradient)
        record = records[call]
        assert record["calls"] == call
        assert record["updates"] == call - 1
        if call == 1:
            np.testing.assert_array_equal(record["params"], initial)
            if rectangular:
                # A rectangular first gradient has a degenerate covariance nullspace.
                for block, group in zip(blocks, record["groups"]):
                    if block.precondition:
                        block.ql = matrix(group, "q_left", (block.shape[0],) * 2)
                        block.qr = matrix(group, "q_right", (block.shape[1],) * 2)
        observed = np.asarray(record["params"])
        parameter_error = max(parameter_error, float(np.max(np.abs(observed - params))))
        np.testing.assert_allclose(observed, params, rtol=2e-8, atol=2e-9)
        for block, group in zip(blocks, record["groups"]):
            assert group["name"] == block.name
            assert group["indices"] == block.indices.tolist()
            actual_m = matrix(group, "m", block.shape)
            actual_v = matrix(group, "v", block.shape)
            assert np.all(np.isfinite(actual_v)) and np.min(actual_v) >= 0
            if block.precondition:
                rows, cols = block.shape
                ql = matrix(group, "q_left", (rows, rows))
                qr = matrix(group, "q_right", (cols, cols))
                actual_m = ql @ actual_m @ qr.T
                for q in (ql, qr):
                    err = np.max(np.abs(q.T @ q - np.eye(q.shape[0])))
                    orthogonality_error = max(orthogonality_error, float(err))
                    np.testing.assert_allclose(q.T @ q, np.eye(q.shape[0]), atol=2e-10)
                for key, expected in (("gg_left", block.left), ("gg_right", block.right)):
                    actual = matrix(group, key, expected.shape)
                    covariance_error = max(covariance_error, float(np.max(np.abs(actual - expected))))
                    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-10)
            expected_m = block.back(block.m)
            moment_error = max(moment_error, float(np.max(np.abs(actual_m - expected_m))))
            np.testing.assert_allclose(actual_m, expected_m, rtol=2e-8, atol=2e-9)
    return {
        "shape_case": "rectangular_nep" if rectangular else "square_full_rank",
        "initial_basis_seeded_from_cpp": rectangular,
        "calls": len(gradients), "updates": len(gradients) - 1,
        "frequency": frequency, "weight_decay": decay,
        "max_parameter_error": parameter_error, "max_physical_moment_error": moment_error,
        "max_covariance_error": covariance_error, "max_orthogonality_error": orthogonality_error,
        "passed": True,
    }


def verify_basis_invariance():
    initial, gradients, blocks, _ = deterministic_case()
    params = initial.copy()
    for gradient in gradients[:13]:
        for block in blocks:
            block.update(params, gradient)
    altered = copy.deepcopy(blocks)
    rng = np.random.default_rng(712)
    for block in altered:
        if not block.precondition:
            continue
        rows, cols = block.shape
        il, ir = rng.permutation(rows), rng.permutation(cols)
        sl, sr = rng.choice([-1, 1], rows), rng.choice([-1, 1], cols)
        block.ql = block.ql[:, il] * sl
        block.qr = block.qr[:, ir] * sr
        block.m = block.m[np.ix_(il, ir)] * sl[:, None] * sr[None, :]
        block.v = block.v[np.ix_(il, ir)]
    changed = params.copy()
    maximum = 0.0
    for gradient in gradients[13:]:
        for block in blocks:
            block.update(params, gradient)
        for block in altered:
            block.update(changed, gradient)
        maximum = max(maximum, float(np.max(np.abs(params - changed))))
        np.testing.assert_allclose(params, changed, rtol=2e-9, atol=2e-10)
    return {"passed": True, "max_parameter_error": maximum, "remaining_calls": len(gradients) - 13}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {
        "reference": "Independent NumPy double-precision SOAP recurrence",
        "upstream_commit": "a1e553530fde97d0e6b307d7c82ac6d38b072340",
        "basis_sign_permutation_invariance": verify_basis_invariance(),
        "cpp_tests": [],
    }
    if args.executable:
        executable = args.executable.resolve()
        for rectangular, decay, frequency in ((False, 0.0, 10), (False, 0.01, 3),
                                               (True, 0.0, 10)):
            report["cpp_tests"].append(verify_trace(executable, rectangular, decay, frequency))
    report["passed"] = True
    content = json.dumps(report, indent=2) + "\n"
    print(content, end="")
    if args.output:
        args.output.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()

"""Run: FCVP_BRIDGE=... python -m unittest discover -s tests/fcvp -v"""

import copy
import os
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/fcvp"))
from reference import (
    Model,
    Oracle,
    read_xyz,
    groups_from_basis,
    objective,
    reference_mm,
)


class FCVPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.bridge = Path(os.environ.get("FCVP_BRIDGE", ROOT / "local/fcvp/libfcvp.so"))
        if not cls.bridge.exists():
            raise RuntimeError(
                "Build the NEP_CPU bridge first; see tools/fcvp/README.md"
            )
        cls.path = ROOT / "examples/nep_train/nep.txt"
        cls.oracle = Oracle(cls.bridge, cls.path)
        cls.model = cls.oracle.model
        cls.frame = read_xyz(ROOT / "examples/nep_train/train.xyz")[0]
        cls.a, cls.offset = cls.oracle.basis(cls.frame)

    @classmethod
    def tearDownClass(cls):
        cls.oracle.close()
        cls.tmp.cleanup()

    def test_actual_nep4_energy_force_virial_linearity(self):
        rng = np.random.default_rng(721)
        for beta in (
            self.model.beta,
            rng.normal(size=len(self.model.beta)),
            np.zeros(len(self.model.beta)),
        ):
            np.testing.assert_allclose(
                self.a @ beta + self.offset,
                self.oracle.predict(self.frame, beta),
                atol=1e-10,
                rtol=1e-10,
            )

    def test_force_includes_other_site_energies(self):
        frame = copy.deepcopy(self.frame)
        n = len(frame.species)
        h = 1e-5
        beta = self.model.beta
        for atom, axis in [(0, 0), (1, 2), (9, 1)]:
            frame.positions[atom, axis] += h
            plus = self.oracle.predict(frame, beta)[0] * n
            frame.positions[atom, axis] -= 2 * h
            minus = self.oracle.predict(frame, beta)[0] * n
            frame.positions[atom, axis] += h
            force = self.oracle.predict(frame, beta)[1 + axis * n + atom]
            self.assertAlmostEqual(force, -(plus - minus) / (2 * h), delta=5e-7)

    def test_virial_is_strain_derivative(self):
        beta = self.model.beta
        h = 1e-6
        base = self.frame
        native = self.oracle.predict(base, beta)[-6:]
        for component, (i, j) in enumerate(
            [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (2, 0)]
        ):
            energies = []
            for sign in (1, -1):
                f = copy.deepcopy(base)
                strain = np.eye(3)
                strain[i, j] += sign * h
                f.positions = f.positions @ strain.T
                f.cell = f.cell @ strain.T
                energies.append(self.oracle.predict(f, beta)[0])
            self.assertAlmostEqual(
                native[component], -(energies[0] - energies[1]) / (2 * h), delta=5e-7
            )

    def test_shared_bias_has_no_force_or_virial(self):
        np.testing.assert_allclose(self.a[1:, -1], 0, atol=1e-13)
        self.assertAlmostEqual(self.a[0, -1], -1.0)

    def test_zbl_is_a_fixed_affine_offset(self):
        header = self.model.header.copy()
        header[0] = header[0].replace("nep4", "nep4_zbl")
        header.insert(1, "zbl 0.5 4.0")
        path = Path(self.tmp.name) / "zbl.txt"
        path.write_text(
            "\n".join(header) + "\n" + "\n".join(map(str, self.model.values)) + "\n"
        )
        other = Oracle(self.bridge, path)
        try:
            a, offset = other.basis(self.frame)
            self.assertGreater(np.linalg.norm(offset), 1e-3)
            np.testing.assert_allclose(
                a @ self.model.beta + offset,
                other.predict(self.frame, self.model.beta),
                atol=1e-9,
                rtol=1e-10,
            )
        finally:
            other.close()

    def test_roundtrip_only_changes_output(self):
        path = Path(self.tmp.name) / "updated.txt"
        beta = self.model.beta * 0.7
        self.model.write(path, beta)
        loaded = Model.read(path)
        mask = np.ones(len(loaded.values), bool)
        mask[self.model.indices] = False
        np.testing.assert_array_equal(loaded.values[mask], self.model.values[mask])
        other = Oracle(self.bridge, path)
        try:
            np.testing.assert_allclose(
                other.predict(self.frame, beta), self.a @ beta + self.offset, atol=1e-10
            )
        finally:
            other.close()

    def test_nep5_type_bias_layout(self):
        m = self.model
        old = m.values
        p = len(m.beta)
        types = len(m.elements)
        ann_block = (max(m.indices[:-1]) + 1) // types
        chunks = []
        for t in range(types):
            chunks.extend(old[t * ann_block : (t + 1) * ann_block])
            chunks.append(0.0)
        chunks.extend(old[types * ann_block :])
        header = m.header.copy()
        header[0] = header[0].replace("nep4", "nep5")
        path = Path(self.tmp.name) / "nep5.txt"
        path.write_text("\n".join(header) + "\n" + "\n".join(map(str, chunks)) + "\n")
        other = Oracle(self.bridge, path)
        try:
            self.assertEqual(len(other.model.indices), p + types - 1)
            a, b = other.basis(self.frame)
            np.testing.assert_allclose(
                other.predict(self.frame, other.model.beta),
                self.oracle.predict(self.frame, m.beta),
                atol=1e-10,
            )
            rng = np.random.default_rng(93)
            beta = rng.normal(size=len(other.model.beta))
            np.testing.assert_allclose(
                a @ beta + b, other.predict(self.frame, beta), atol=1e-10
            )
            self.assertEqual(
                other.model.training_parameter_count, other.model.parameter_count - 1
            )
            groups = [(rng.normal(size=(20, 3)), rng.normal(size=20), 1.0)]
            x, history = other.solve(
                groups, np.ones(3), iterations=50, lambda1=50.0, lambda2=20.0
            )
            ref, hr = reference_mm(
                groups,
                np.ones(3),
                iterations=50,
                lambda1=50.0,
                lambda2=20.0,
                fixed=other.model.fixed_regularization_parameters(),
                total=other.model.training_parameter_count,
            )
            self.assertAlmostEqual(history[-1], hr[-1], delta=2e-7)
        finally:
            other.close()

    def test_mm_agrees_with_independent_svd_reference(self):
        rng = np.random.default_rng(54)
        p = 6
        groups = [
            (rng.normal(size=(n, p)), rng.normal(size=n), w)
            for n, w in [(25, 1), (49, 2), (18, 0.1)]
        ]
        beta = rng.normal(size=p)
        x, history = self.oracle.solve(groups, beta, iterations=150)
        ref, _ = reference_mm(groups, beta, iterations=150)
        self.assertTrue(np.all(np.diff(history) <= 0))
        np.testing.assert_allclose(x, ref, atol=2e-6, rtol=2e-5)
        self.assertLess(objective(groups, x), objective(groups, beta))

    def test_l1_l2_include_fixed_nonlinear_parameters(self):
        rng = np.random.default_rng(126)
        p = 4
        groups = [(rng.normal(size=(30, p)), rng.normal(size=30), 1.0)]
        beta = rng.normal(size=p)
        l1 = 200.0
        l2 = 20.0
        fixed = self.model.values[: self.model.parameter_count].copy()
        fixed[self.model.indices] = 0
        x, history = self.oracle.solve(
            groups, beta, iterations=180, lambda1=l1, lambda2=l2
        )
        ref, hr = reference_mm(
            groups,
            beta,
            iterations=180,
            lambda1=l1,
            lambda2=l2,
            fixed=fixed,
            total=self.model.parameter_count,
        )
        opts = dict(
            lambda1=l1, lambda2=l2, fixed=fixed, total=self.model.parameter_count
        )
        self.assertTrue(np.all(np.diff(history) <= 0))
        self.assertAlmostEqual(
            objective(groups, x, **opts), objective(groups, ref, **opts), delta=2e-7
        )

    def test_rank_deficiency_and_zero_lambda(self):
        rng = np.random.default_rng(313)
        a = rng.normal(size=(30, 3))
        a = np.column_stack([a, a[:, 0], np.zeros(30)])
        y = rng.normal(size=30)
        groups = [(a, y, 1.0), (a, 10 * y, 0.0)]
        beta = np.ones(5)
        x, history = self.oracle.solve(groups, beta, iterations=50)
        ref, _ = reference_mm(groups, beta, iterations=50)
        self.assertTrue(np.isfinite(x).all())
        self.assertTrue(np.all(np.diff(history) <= 0))
        np.testing.assert_allclose(a @ x, a @ ref, atol=1e-8)

    def test_exact_fit_and_missing_virial(self):
        rng = np.random.default_rng(14)
        a = rng.normal(size=(15, 4))
        beta = rng.normal(size=4)
        x, history = self.oracle.solve([(a, a @ beta, 1)], beta, iterations=15)
        self.assertTrue(np.isfinite(history).all())
        np.testing.assert_allclose(a @ x, a @ beta, atol=1e-8)
        self.assertEqual(
            len(groups_from_basis([self.frame], [(self.a, self.offset)])), 2
        )

    def test_nonfinite_options_rejected(self):
        a = np.ones((4, 2))
        y = np.ones(4)
        with self.assertRaises(RuntimeError):
            self.oracle.solve([(a, y, 1)], np.zeros(2), epsilon=float("nan"))

    def test_xyz_rejects_partial_pbc_and_preserves_weights(self):
        f = copy.deepcopy(self.frame)
        f.weight = 2.0
        f.virial = np.arange(6.0)
        a = self.a
        groups = groups_from_basis(
            [f],
            [(a, self.offset)],
            type_weights={"Pb": 3.0, "Te": 2.0},
            force_delta=0.2,
            shear=4.0,
        )
        np.testing.assert_allclose(groups[0][0], 2 * a[:1])
        np.testing.assert_allclose(
            groups[2][0], a[-6:] * np.array([2, 2, 2, 8, 8, 8])[:, None]
        )
        path = Path(self.tmp.name) / "bad.xyz"
        path.write_text(f.text.replace("T T T", "T F T"))
        with self.assertRaises(ValueError):
            read_xyz(path)


if __name__ == "__main__":
    unittest.main()

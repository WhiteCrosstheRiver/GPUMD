"""Optional native NEP5 FCVP generality tests.

Set FCVP_GNEP for GPU checks, FCVP_BRIDGE for independent NEP_CPU checks,
and FCVP_DEVICES=0,1 for the two-device comparison.
"""

import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/fcvp"))
from reference import Model, Oracle, read_xyz


CELL = 'Lattice="40 0 0 0 40 0 0 0 40" '
PROPERTIES = "Properties=species:S:1:pos:R:3:force:R:3\n"
VIRIAL = 'virial="1 .2 -.3 .2 2 .4 -.3 .4 3" '
FRAMES = (
    "4\n" + CELL + "energy=-2 weight=1.3 " + VIRIAL + PROPERTIES
    + "Si 2 2 2 .1 .2 -.1\nGe 3.7 2.2 2.3 -.2 .1 .3\n"
    + "Si 2.5 3.8 2.4 .2 -.1 .1\nGe 3.1 2.7 4.1 -.1 -.2 -.3\n",
    "3\n" + CELL + "energy=-1 weight=.7 " + PROPERTIES
    + "Ge 8 8 8 .1 -.1 .2\nSi 9.5 8.7 8.3 -.1 .1 -.2\n"
    + "Ge 8.4 9.6 8.9 .2 .1 -.1\n",
    "2\n" + CELL + "energy=.2 weight=.9 " + VIRIAL + PROPERTIES
    + "Si 15 15 15 0 0 0\nGe 17 16 15.5 .1 -.1 .2\n",
)
BASE = (
    "type 2 Si Ge\ncutoff 5 4\nn_max 2 2\nbasis_size 3 3\n"
    "neuron 8\nbatch 3\nepoch 1\nweight_decay 0\n"
)
LOSS = "lambda_shear 1.5\ntype_weight 1.2 .8\nforce_delta .2\n"
FLEXIBLE_ZBL = (
    # Si-Si, Si-Ge, Ge-Ge. Four positive screening terms sum to about one.
    "1.2 2.4 .22 3.2 .4717 .9423 .2802 .4029 .02817 .2016\n"
    "1.2 2.4 .24 3.2 .4517 .9423 .2802 .4029 .02817 .2016\n"
    "1.2 2.4 .26 3.2 .4317 .9423 .2802 .4029 .02817 .2016\n"
)


def zbl_pair(radius, z1, z2, coefficients, inner, outer):
    """Independent pair energy and radial derivative for the ZBL cutoff."""
    inverse_screening_length = (z1 ** .23 + z2 ** .23) * 2.134563
    x = radius * inverse_screening_length
    terms = [a * np.exp(-b * x) for a, b in zip(coefficients[::2], coefficients[1::2])]
    phi = sum(terms)
    phi_prime = -sum(b * term for b, term in zip(coefficients[1::2], terms))
    if radius < inner:
        cutoff, cutoff_prime = 1.0, 0.0
    elif radius < outer:
        phase = np.pi * (radius - inner) / (outer - inner)
        cutoff = (1 + np.cos(phase)) / 2
        cutoff_prime = -np.pi * np.sin(phase) / (2 * (outer - inner))
    else:
        cutoff, cutoff_prime = 0.0, 0.0
    prefactor = 14.399645 * z1 * z2
    energy = prefactor * phi * cutoff / radius
    derivative = prefactor * (
        (phi_prime * inverse_screening_length / radius - phi / radius**2) * cutoff
        + phi * cutoff_prime / radius
    )
    return energy, derivative


def run_case(root, name, config, frames=FRAMES, model=None, zbl=None, devices=None):
    directory = root / name
    directory.mkdir()
    for filename in ("train.xyz", "test.xyz"):
        (directory / filename).write_text("".join(frames))
    (directory / "gnep.in").write_text(config)
    if model is not None:
        (directory / "nep.txt").write_text(model)
    if zbl is not None:
        (directory / "zbl.in").write_text(zbl)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = devices or os.environ.get("FCVP_DEVICE", "0")
    result = subprocess.run(
        [str(Path(os.environ["FCVP_GNEP"]).resolve())],
        cwd=directory, env=env, capture_output=True, text=True, timeout=240,
    )
    (directory / "stdout.log").write_text(result.stdout + result.stderr)
    return result, directory


def predicted(directory):
    return {
        name: np.loadtxt(directory / f"{name}_train.out", ndmin=2)
        for name in ("energy", "force", "virial")
    }


@unittest.skipUnless(os.environ.get("FCVP_GNEP"), "Set FCVP_GNEP")
class HigherBodyNativeTests(unittest.TestCase):
    def assert_success(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_lmax_matrix_checks_new_descriptor_gradients(self):
        # With n_max_angular=2 and neuron=8, seed chooses a first-layer
        # coefficient in the new 4-body or 5-body descriptor block for the
        # native finite-difference gate's first sampled parameter.
        for lmax, seed, expected in (
            ("4", 31, (4, 0, 0)),
            ("4 2 0", 136, (4, 2, 0)),
            ("4 2 1", 184, (4, 2, 1)),
        ):
            with self.subTest(lmax=lmax), tempfile.TemporaryDirectory() as temp:
                config = (
                    BASE + LOSS + f"l_max {lmax}\nseed {seed}\n"
                    "loss_rmse 1e-3\nfcvp_check 1\n"
                )
                result, directory = run_case(Path(temp), "gradient", config)
                self.assert_success(result)
                self.assertIn("analytical gradient gate passed", result.stdout)
                model = Model.read(directory / "nep.txt")
                self.assertEqual(
                    tuple(map(int, next(
                        line.split()[1:] for line in model.header
                        if line.startswith("l_max ")
                    ))), expected,
                )
                self.assertIn(f"gradient check parameter {seed} ", result.stdout)

    def test_invalid_higher_body_combinations_fail(self):
        for lmax in ("4 0 1", "1 2 0", "4 1 0", "4 2 2"):
            with self.subTest(lmax=lmax), tempfile.TemporaryDirectory() as temp:
                result, _ = run_case(
                    Path(temp), "invalid", BASE + f"l_max {lmax}\n"
                )
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("l_max", result.stdout + result.stderr)

    def test_maximum_descriptor_dimension_220(self):
        config = (
            BASE.replace("n_max 2 2", "n_max 19 19")
            .replace("basis_size 3 3", "basis_size 19 19")
            .replace("neuron 8", "neuron 2")
            + "l_max 8 2 1\nseed 405\nloss_rmse 1e-3\nfcvp_check 1\n"
        )
        with tempfile.TemporaryDirectory() as temp:
            result, directory = run_case(Path(temp), "wide_descriptor", config)
            self.assert_success(result)
            self.assertIn("analytical gradient gate passed", result.stdout)
            self.assertIn("gradient check parameter 405 ", result.stdout)
            model = Model.read(directory / "nep.txt")
            self.assertEqual(len(model.values) - model.parameter_count, 220)

    def test_exhaustive_higher_body_parameter_gradients(self):
        for lmax, num_angular_channels in (("4 2 0", 5), ("4 2 1", 6)):
            with self.subTest(lmax=lmax), tempfile.TemporaryDirectory() as temp:
                config = (
                    BASE.replace("n_max 2 2", "n_max 1 1")
                    .replace("basis_size 3 3", "basis_size 2 2")
                    .replace("neuron 8", "neuron 2")
                    + LOSS + f"l_max {lmax}\nseed 41\n"
                    "lambda_e .7\nlambda_f 1.2\nlambda_v .3\n"
                    "loss_rmse 1e-3\nfcvp_check 2\n"
                )
                result, directory = run_case(Path(temp), "exhaustive", config)
                self.assert_success(result)
                self.assertIn("analytical gradient gate passed", result.stdout)
                dim = 2 + 2 * num_angular_channels
                expected_count = 2 * ((dim + 2) * 2 + 1) + 4 * (2 * 3 + 2 * 3)
                model = Model.read(directory / "nep.txt")
                self.assertEqual(model.parameter_count - 1, expected_count)
                checked = [int(index) for index in re.findall(
                    r"FCVP gradient check parameter (\d+) analytical", result.stdout
                )]
                self.assertEqual(len(checked), expected_count)
                self.assertEqual(sorted(checked), list(range(expected_count)))

    @unittest.skipUnless(os.environ.get("FCVP_BRIDGE"), "Set FCVP_BRIDGE")
    def test_exported_higher_body_models_match_independent_cpu_efv(self):
        bridge = Path(os.environ["FCVP_BRIDGE"])
        for lmax in ("4", "4 2 0", "4 2 1"):
            with self.subTest(lmax=lmax), tempfile.TemporaryDirectory() as temp:
                result, trained = run_case(
                    Path(temp), "train",
                    BASE + LOSS + f"l_max {lmax}\nseed 20261015\n"
                    "fcvp 1 20 1e-3 1e-8\nlambda_2 .1\n",
                )
                self.assert_success(result)
                model_path = trained / "nep.txt"
                model = Model.read(model_path)
                frames = read_xyz(trained / "train.xyz")
                oracle = Oracle(bridge, model_path)
                try:
                    expected = [oracle.predict(frame, model.beta) for frame in frames]
                    for frame, value in zip(frames, expected):
                        basis, offset = oracle.basis(frame)
                        np.testing.assert_allclose(
                            basis @ model.beta + offset, value,
                            rtol=1e-10, atol=1e-9,
                        )
                finally:
                    oracle.close()
                result, directory = run_case(
                    Path(temp), "predict", "prediction 1\nbatch 2\n",
                    model=model_path.read_text(),
                )
                self.assert_success(result)
                actual = predicted(directory)
                self.assertEqual(actual["energy"].shape, (3, 2))
                self.assertEqual(actual["force"].shape, (9, 6))
                self.assertEqual(actual["virial"].shape, (3, 12))
                np.testing.assert_allclose(
                    actual["energy"][:, 0], [row[0] for row in expected],
                    rtol=8e-4, atol=2e-4,
                )
                np.testing.assert_allclose(
                    actual["force"][:, :3],
                    np.concatenate([
                        row[1:1 + 3 * len(frame.species)].reshape(3, -1).T
                        for frame, row in zip(frames, expected)
                    ]), rtol=2e-3, atol=2e-3,
                )
                np.testing.assert_allclose(
                    actual["virial"][:, :6], [row[-6:] for row in expected],
                    rtol=2e-3, atol=2e-3,
                )

    @unittest.skipUnless(os.environ.get("FCVP_BRIDGE"), "Set FCVP_BRIDGE")
    def test_fixed_flexible_zbl_offset_and_native_prediction(self):
        close = (
            "3\n" + CELL + "energy=0 " + VIRIAL + PROPERTIES
            + "Si 2 2 2 0 0 0\nGe 2.95 2.18 2.12 0 0 0\n"
            + "Si 2.3 3.3 2.5 0 0 0\n"
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = (
                BASE + "l_max 4 2 1\nseed 20261015\nzbl 2.4\n"
                "fcvp 1 10 1e-2 1e-7\nlambda_2 .5\n"
                "lambda_f 0\nlambda_v 0\n"
            )
            result, trained = run_case(root, "train", config, (close,), zbl=FLEXIBLE_ZBL)
            self.assert_success(result)
            model_path = trained / "nep.txt"
            model = Model.read(model_path)
            self.assertEqual(model.header[0].split()[0], "nep5_zbl")
            self.assertIn("zbl 0 0", model.header)
            np.testing.assert_allclose(model.values[-30:], np.fromstring(FLEXIBLE_ZBL, sep=" "), rtol=2e-7)
            frame = read_xyz(trained / "train.xyz")[0]
            oracle = Oracle(Path(os.environ["FCVP_BRIDGE"]), model_path)
            try:
                basis, offset = oracle.basis(frame)
                cpu = oracle.predict(frame, model.beta)
                np.testing.assert_allclose(basis @ model.beta + offset, cpu, rtol=1e-10, atol=1e-8)
            finally:
                oracle.close()
            no_zbl_path = root / "no_zbl.txt"
            plain = Model(
                [line.replace("nep5_zbl", "nep5") for line in model.header if not line.startswith("zbl ")],
                model.values[:-30], model.indices, model.parameter_count,
                model.elements, model.shared_bias_index,
            )
            plain.write(no_zbl_path, plain.beta)
            no_zbl = Oracle(Path(os.environ["FCVP_BRIDGE"]), no_zbl_path)
            try:
                plain_basis, plain_offset = no_zbl.basis(frame)
                np.testing.assert_allclose(basis, plain_basis, rtol=1e-10, atol=1e-9)
                self.assertGreater(np.linalg.norm(offset - plain_offset), 1.0)
                self.assertGreater(np.linalg.norm((offset - plain_offset)[1:-6]), 1.0)
                self.assertGreater(np.linalg.norm((offset - plain_offset)[-6:]), .01)
            finally:
                no_zbl.close()
            result, directory = run_case(
                root, "predict", "prediction 1\nbatch 1\n", (close,),
                model=model_path.read_text(),
            )
            self.assert_success(result)
            actual = predicted(directory)
            np.testing.assert_allclose(actual["energy"][0, 0], cpu[0], rtol=2e-3, atol=2e-3)
            np.testing.assert_allclose(actual["force"][:, :3], cpu[1:10].reshape(3, 3).T,
                                       rtol=2e-3, atol=2e-2)
            np.testing.assert_allclose(actual["virial"][0, :6], cpu[-6:],
                                       rtol=2e-3, atol=2e-2)

    @unittest.skipUnless(os.environ.get("FCVP_BRIDGE"), "Set FCVP_BRIDGE")
    def test_prediction_folds_nonzero_shared_bias_into_type_biases(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result, trained = run_case(
                root, "train", BASE + "l_max 4 2 1\nseed 73\n",
            )
            self.assert_success(result)
            model = Model.read(trained / "nep.txt")
            self.assertIsNotNone(model.shared_bias_index)
            frames = read_xyz(trained / "train.xyz")
            original_oracle = Oracle(Path(os.environ["FCVP_BRIDGE"]), trained / "nep.txt")
            try:
                original = [original_oracle.predict(frame, model.beta) for frame in frames]
            finally:
                original_oracle.close()
            model.values[model.shared_bias_index] = .375
            biased_path = root / "biased.txt"
            model.write(biased_path, model.beta)
            oracle = Oracle(Path(os.environ["FCVP_BRIDGE"]), biased_path)
            try:
                expected = [oracle.predict(frame, model.beta) for frame in frames]
            finally:
                oracle.close()
            for before, after in zip(original, expected):
                self.assertAlmostEqual(before[0] - after[0], .375, delta=1e-9)
                np.testing.assert_allclose(before[1:], after[1:], rtol=1e-10, atol=1e-9)
            result, directory = run_case(
                root, "biased_prediction", "prediction 1\nbatch 2\n",
                model=biased_path.read_text(),
            )
            self.assert_success(result)
            actual = predicted(directory)
            np.testing.assert_allclose(actual["energy"][:, 0],
                                       [row[0] for row in expected],
                                       rtol=8e-4, atol=2e-4)
            np.testing.assert_allclose(actual["force"][:, :3],
                                       np.concatenate([
                                           row[1:1 + 3 * len(frame.species)].reshape(3, -1).T
                                           for frame, row in zip(frames, expected)
                                       ]), rtol=2e-3, atol=2e-3)
            np.testing.assert_allclose(actual["virial"][:, :6],
                                       [row[-6:] for row in expected],
                                       rtol=2e-3, atol=2e-3)

    def test_prediction_rejects_truncated_and_nep4_models(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result, trained = run_case(
                root, "train", BASE + "l_max 4 2 1\nseed 73\n",
            )
            self.assert_success(result)
            model = (trained / "nep.txt").read_text()
            cases = (
                ("truncated", "\n".join(model.splitlines()[:-1]) + "\n", None),
                ("nep4", model.replace("nep5 ", "nep4 ", 1), "nep5"),
            )
            for name, contents, message in cases:
                with self.subTest(name=name):
                    result, _ = run_case(
                        root, name, "prediction 1\n", model=contents,
                    )
                    self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                    if message:
                        self.assertIn(message, (result.stdout + result.stderr).lower())

    def test_prediction_uses_model_elements_despite_conflicting_input(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result, trained = run_case(
                root, "train", BASE + "l_max 4 2 1\nseed 73\n",
            )
            self.assert_success(result)
            model = (trained / "nep.txt").read_text()
            outputs = []
            for name, config in (
                ("model_only", "prediction 1\nbatch 2\n"),
                (
                    "contradictory",
                    "prediction 1\nbatch 2\ntype 2 Ge Si\n"
                    "cutoff 9 5\nn_max 0 0\nbasis_size 0 0\n"
                    "l_max 0\nneuron 1\n",
                ),
            ):
                result, directory = run_case(root, name, config, model=model)
                self.assert_success(result)
                outputs.append(predicted(directory))
            for name in outputs[0]:
                np.testing.assert_allclose(outputs[0][name], outputs[1][name],
                                           rtol=1e-6, atol=1e-6)

    def test_universal_and_large_flexible_zbl_survive_typewise_nep_cutoff(self):
        universal = (.18175, 3.1998, .50986, .94229,
                     .28022, .4029, .02817, .20162)
        flexible = (.22, 3.1998, .47161, .94229,
                    .28022, .4029, .02817, .20162)
        cases = (
            ("universal", "H", "H", 1, 1, "type 1 H\n", None, universal),
            (
                "flexible_11_types", "H", "He", 1, 2,
                "type 11 H He Li Be B C N O F Ne Na\n",
                "1.25 2.5 " + " ".join(map(str, flexible)) + "\n",
                flexible,
            ),
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name, first, second, z1, z2, type_line, pair_line, coefficients in cases:
                with self.subTest(name=name):
                    frame = (
                        "2\n" + CELL + "energy=0 " + VIRIAL + PROPERTIES
                        + f"{first} 2 2 2 0 0 0\n"
                        + f"{second} 3.5 2 2 0 0 0\n"
                    )
                    zbl = None if pair_line is None else pair_line * 66
                    config = (
                        type_line + "cutoff 5 3\nn_max 0 0\nbasis_size 0 0\n"
                        "l_max 0\nneuron 2\nbatch 1\nepoch 1\nseed 19\n"
                        "use_typewise_cutoff 1.5 1.5\nzbl 2.5\n"
                        "lambda_e 0\nlambda_f 0\nlambda_v 0\n"
                    )
                    result, trained = run_case(root, name + "_train", config, (frame,), zbl=zbl)
                    self.assert_success(result)
                    model = Model.read(trained / "nep.txt")
                    zero_path = root / (name + "_zero.txt")
                    model.write(zero_path, np.zeros_like(model.beta))
                    result, directory = run_case(
                        root, name + "_predict", "prediction 1\n", (frame,),
                        model=zero_path.read_text(),
                    )
                    self.assert_success(result)
                    expected_energy, derivative = zbl_pair(
                        1.5, z1, z2, coefficients, 1.25, 2.5,
                    )
                    self.assertGreater(expected_energy, 0.01)
                    actual = predicted(directory)
                    np.testing.assert_allclose(actual["energy"][0, 0], expected_energy / 2,
                                               rtol=3e-3, atol=2e-4)
                    np.testing.assert_allclose(actual["force"][:, :3],
                                               [[derivative, 0, 0], [-derivative, 0, 0]],
                                               rtol=3e-3, atol=2e-4)
                    np.testing.assert_allclose(actual["virial"][0, :6],
                                               [-1.5 * derivative / 2, 0, 0, 0, 0, 0],
                                               rtol=3e-3, atol=2e-4)

    @unittest.skipUnless(os.environ.get("FCVP_DEVICES"), "Set FCVP_DEVICES=0,1")
    def test_higher_body_one_and_two_gpu_agree_with_missing_virial(self):
        devices = os.environ["FCVP_DEVICES"]
        self.assertEqual(len(devices.split(",")), 2)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = (
                BASE + LOSS + "l_max 4 2 1\nseed 20261015\n"
                "fcvp 1 20 1e-2 1e-8\nlambda_2 .5\nfcvp_check 1\n"
            )
            outputs = []
            for label, visible in (("one", devices.split(",")[0]), ("two", devices)):
                result, directory = run_case(root, label, config, devices=visible)
                self.assert_success(result)
                self.assertIn("analytical gradient gate passed", result.stdout)
                outputs.append((
                    np.loadtxt(directory / "gnep.restart"),
                    np.loadtxt(directory / "fcvp.out", ndmin=2)[:, :3],
                ))
            for one, two in zip(*outputs):
                np.testing.assert_allclose(one, two, rtol=4e-4, atol=4e-5)

    def test_output_parameter_count_can_exceed_512(self):
        frame = (
            "3\n" + CELL + "energy=0 " + PROPERTIES
            + "Si 1 1 1 0 0 0\nGe 10 10 10 0 0 0\nC 20 20 20 0 0 0\n"
        )
        config = (
            "type 3 Si Ge C\ncutoff 5 4\nn_max 0 0\nbasis_size 0 0\n"
            "l_max 0\nneuron 180\nbatch 1\nepoch 1\nseed 23\n"
            "weight_decay 0\nfcvp 1 2 1e-2 1e-4\nlambda_2 1\n"
        )
        with tempfile.TemporaryDirectory() as temp:
            result, directory = run_case(Path(temp), "wide", config, (frame,))
            self.assert_success(result)
            self.assertEqual(len(Model.read(directory / "nep.txt").beta), 543)
            self.assertTrue(np.isfinite(np.loadtxt(directory / "fcvp.out", ndmin=2)).all())

    def test_small_basis_budget_matches_default(self):
        # Four sparse 100-atom frames plus three Gram matrices exceed 1 MiB
        # with 93 output parameters, while a one-frame tile fits.
        atoms = [f"{('Si', 'Ge', 'C')[i % 3]} {2 + 6 * (i % 10)} {2 + 6 * ((i // 10) % 10)} 2 0 0 0\n"
                 for i in range(100)]
        frame = (
            "100\n" + 'Lattice="80 0 0 0 80 0 0 0 80" '
            + "energy=0 " + PROPERTIES + "".join(atoms)
        )
        config = (
            "type 3 Si Ge C\ncutoff 5 4\nn_max 0 0\nbasis_size 0 0\n"
            "l_max 0\nneuron 30\nbatch 4\nepoch 1\nseed 29\n"
            "weight_decay 0\nfcvp 1 5 1e-2 1e-4\nlambda_2 1\n"
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            outputs = []
            for limit in (1, 512):
                result, directory = run_case(
                    root, str(limit), config + f"fcvp_memory {limit}\n",
                    (frame,) * 4,
                )
                self.assert_success(result)
                outputs.append((
                    np.loadtxt(directory / "gnep.restart"),
                    np.loadtxt(directory / "fcvp.out", ndmin=2)[:, :3],
                    np.loadtxt(directory / "fcvp.out", ndmin=2)[:, 7],
                ))
            for one, other in zip(outputs[0][:2], outputs[1][:2]):
                np.testing.assert_allclose(one, other, rtol=4e-4, atol=4e-5)
            self.assertLess(outputs[0][2].max(), outputs[1][2].max())


if __name__ == "__main__":
    unittest.main()

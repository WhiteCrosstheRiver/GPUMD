"""Optional native tests; set FCVP_GNEP and FCVP_DEVICES=0,1 for two GPUs."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np


FRAMES = (
    '3\nLattice="30 0 0 0 30 0 0 0 30" energy=-2 weight=1.3 '
    'virial="1 .2 .3 .2 2 .4 .3 .4 3" '
    "Properties=species:S:1:pos:R:3:force:R:3\n"
    "Si 1 1 1 .1 .2 -.1\nGe 3.2 2.1 1.8 -.2 .1 .3\nSi 16 16 16 0 0 0\n",
    '2\nLattice="30 0 0 0 30 0 0 0 30" energy=-1 weight=.7 '
    "Properties=species:S:1:pos:R:3:force:R:3\n"
    "Ge 1 1 1 .1 -.1 .2\nSi 3.4 1.8 2.2 -.1 .1 -.2\n",
    '1\nLattice="30 0 0 0 30 0 0 0 30" energy=.2 weight=.9 '
    'virial="0 0 0 0 0 0 0 0 0" '
    "Properties=species:S:1:pos:R:3:force:R:3\nGe 1 1 1 0 0 0\n",
)
COMMON = (
    "type 2 Si Ge\ncutoff 5 4\nn_max 4 4\nbasis_size 8 8\n"
    "neuron 8\nl_max 4\nbatch 3\nepoch 2\nweight_decay 0\n"
)
WEIGHTS = "lambda_shear 2\ntype_weight 1.3 .7\nforce_delta .2\n"


@unittest.skipUnless(os.environ.get("FCVP_GNEP"), "Set FCVP_GNEP for native GPU tests")
class GPUIntegrationTests(unittest.TestCase):
    def run_case(
        self,
        root,
        name,
        extra="",
        devices=None,
        frames=FRAMES,
        binary=None,
        model=None,
    ):
        directory = root / name
        directory.mkdir()
        for filename in ("train.xyz", "test.xyz"):
            (directory / filename).write_text("".join(frames))
        (directory / "gnep.in").write_text(COMMON + extra)
        if model is not None:
            (directory / "nep.txt").write_text(model)
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = devices or os.environ.get(
            "FCVP_DEVICE", "0"
        )
        result = subprocess.run(
            [str(Path(binary or os.environ["FCVP_GNEP"]).resolve())],
            cwd=directory,
            env=environment,
            text=True,
            capture_output=True,
            timeout=180,
        )
        (directory / "stdout.log").write_text(result.stdout + result.stderr)
        return result, directory

    def assert_success(self, result, directory):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(np.isfinite(np.loadtxt(directory / "gnep.restart")).all())
        if (directory / "fcvp.out").exists():
            projection = np.loadtxt(directory / "fcvp.out", ndmin=2)
            self.assertTrue(np.isfinite(projection).all())
            self.assertTrue(np.all(projection[:, 2] <= projection[:, 1] + 1e-8))

    def test_weighted_native_gradients_and_projection(self):
        with tempfile.TemporaryDirectory() as temporary:
            for devices in dict.fromkeys(
                (
                    os.environ.get("FCVP_DEVICE", "0"),
                    os.environ.get("FCVP_DEVICES", "0"),
                )
            ):
                with self.subTest(devices=devices):
                    result, directory = self.run_case(
                        Path(temporary),
                        "weighted_" + devices,
                        WEIGHTS + "seed 2147483647\nfcvp 1 40 1e-4 1e-8\nfcvp_check 1\n"
                        "lambda_1 .05\nlambda_2 .05\nzbl 2.5\n",
                        devices=devices,
                        frames=tuple(
                            frame.replace("3.2 2.1 1.8", "2.8 1.4 1.3")
                            for frame in FRAMES
                        ),
                    )
                    self.assert_success(result, directory)
                    self.assertIn("analytical gradient gate passed", result.stdout)

    def test_isolated_structure_with_more_devices_than_configurations(self):
        with tempfile.TemporaryDirectory() as temporary:
            for devices in dict.fromkeys(
                (
                    os.environ.get("FCVP_DEVICE", "0"),
                    os.environ.get("FCVP_DEVICES", "0"),
                )
            ):
                result, directory = self.run_case(
                    Path(temporary),
                    "isolated_" + devices,
                    "seed 7\nfcvp 1 20 1e-4 1e-8\nfcvp_check 1\n",
                    devices=devices,
                    frames=FRAMES[-1:],
                )
                self.assert_success(result, directory)

    def test_invalid_projection_options_fail_before_training(self):
        cases = (
            ("fcvp 0 40 1e-8 1e-8\n", "fcvp requires"),
            ("fcvp 1 40 nan 1e-8\n", "fcvp requires"),
            ("loss_rmse 1e-300\n", "positive normal float range"),
            ("fcvp 1 40 1e-8 1e-8\nweight_decay .01\n", "weight_decay 0"),
            ("fcvp 1 40 1e-8 1e-8\nbatch 1 1\n", "batch accumulation"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            for index, (extra, message) in enumerate(cases):
                result, _ = self.run_case(Path(temporary), str(index), extra)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stdout + result.stderr)

    @unittest.skipUnless(
        os.environ.get("FCVP_BASELINE_GNEP"), "Set baseline binary for regression"
    )
    def test_default_path_matches_delivery_binary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outputs = []
            for name, binary in (
                ("delivery", os.environ["FCVP_BASELINE_GNEP"]),
                ("integrated", os.environ["FCVP_GNEP"]),
            ):
                result, directory = self.run_case(root, name, binary=binary)
                self.assert_success(result, directory)
                outputs.append(np.loadtxt(directory / "gnep.restart"))
            np.testing.assert_allclose(
                *outputs, rtol=1e-5, atol=2e-6, err_msg="default delivery compatibility"
            )

    @unittest.skipUnless(os.environ.get("FCVP_DEVICES"), "Set FCVP_DEVICES to two GPUs")
    def test_legacy_isolated_multiatom_shard_matches_global_neighbor_branch(self):
        devices = os.environ["FCVP_DEVICES"]
        self.assertGreater(len(devices.split(",")), 1)
        isolated_frame = (
            FRAMES[0]
            .replace("Si 1 1 1", "Si 5 5 5")
            .replace("Ge 3.2 2.1 1.8", "Ge 12 12 12")
            .replace("Si 16 16 16", "Si 20 20 20")
        )
        with tempfile.TemporaryDirectory() as temporary:
            outputs = []
            for suffix, visible in (
                ("single", devices.split(",")[0]),
                ("dual", devices),
            ):
                result, directory = self.run_case(
                    Path(temporary),
                    "legacy_isolated_" + suffix,
                    "seed 20261015\n" + WEIGHTS,
                    devices=visible,
                    frames=(isolated_frame, *FRAMES[1:]),
                )
                self.assert_success(result, directory)
                outputs.append(np.loadtxt(directory / "gnep.restart"))
            np.testing.assert_allclose(
                *outputs,
                rtol=3e-4,
                atol=3e-5,
                err_msg="legacy isolated multiatom shard",
            )

    def test_unregularized_projection_native_gate_and_monotonic_objective(self):
        with tempfile.TemporaryDirectory() as temporary:
            for devices in dict.fromkeys(
                (
                    os.environ.get("FCVP_DEVICE", "0"),
                    os.environ.get("FCVP_DEVICES", "0"),
                )
            ):
                result, directory = self.run_case(
                    Path(temporary),
                    "unregularized_" + devices,
                    "seed 20261015\n" + WEIGHTS + "fcvp 1 40 1e-4 1e-8\nfcvp_check 1\n",
                    devices=devices,
                )
                self.assert_success(result, directory)
                self.assertIn("analytical gradient gate passed", result.stdout)

    @unittest.skipUnless(os.environ.get("FCVP_DEVICES"), "Set FCVP_DEVICES to two GPUs")
    def test_two_gpu_matches_one_gpu_for_losses_and_projection_intervals(self):
        devices = os.environ["FCVP_DEVICES"]
        self.assertGreater(len(devices.split(",")), 1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            # This tiny basis is underdetermined; regularize the parameter-equality
            # cases. Unregularized reconstruction and descent are tested separately.
            for label, extra in (
                ("legacy", ""),
                ("energy_shift", "energy_shift 1\nloss_rmse 1e-4\nfcvp_check 1\n"),
                ("rmse", "loss_rmse 1e-4\nfcvp_check 1\n"),
                ("fcvp1", "fcvp 1 40 1e-2 1e-8\nlambda_2 .05\nfcvp_check 1\n"),
                (
                    "fcvp_batch2",
                    "fcvp 1 40 1e-2 1e-8\nlambda_2 1\nbatch 2\nfcvp_check 1\n",
                ),
                (
                    "fcvp3",
                    "fcvp 3 40 1e-2 1e-8\nlambda_2 .05\nbatch 1\nfcvp_check 1\n",
                ),
            ):
                outputs = []
                objectives = []
                for suffix, visible in (
                    ("single", devices.split(",")[0]),
                    ("dual", devices),
                ):
                    result, directory = self.run_case(
                        root,
                        label + suffix,
                        "seed 20261015\n" + WEIGHTS + extra,
                        devices=visible,
                    )
                    self.assert_success(result, directory)
                    outputs.append(np.loadtxt(directory / "gnep.restart"))
                    if (directory / "fcvp.out").exists():
                        objectives.append(
                            np.loadtxt(directory / "fcvp.out", ndmin=2)[:, :3]
                        )
                np.testing.assert_allclose(
                    *outputs, rtol=3e-4, atol=3e-5, err_msg=label
                )
                if objectives:
                    np.testing.assert_allclose(
                        *objectives, rtol=3e-4, atol=3e-5, err_msg=label
                    )

    @unittest.skipUnless(os.environ.get("FCVP_DEVICES"), "Set FCVP_DEVICES to two GPUs")
    def test_prediction_retains_all_structures_and_atoms_across_shards(self):
        devices = os.environ["FCVP_DEVICES"]
        self.assertGreater(len(devices.split(",")), 1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result, trained = self.run_case(
                root,
                "training",
                "seed 20261015\nfcvp 1 40 1e-4 1e-8\n",
                devices=devices.split(",")[0],
            )
            self.assert_success(result, trained)
            model = (trained / "nep.txt").read_text()
            for batch in (1, 2, 3):
                predictions = []
                for suffix, visible in (
                    ("single", devices.split(",")[0]),
                    ("dual", devices),
                ):
                    result, directory = self.run_case(
                        root,
                        f"prediction_{batch}_{suffix}",
                        f"prediction 1\nbatch {batch}\n",
                        devices=visible,
                        model=model,
                    )
                    self.assertEqual(
                        result.returncode, 0, result.stdout + result.stderr
                    )
                    outputs = {}
                    for name, shape in (
                        ("energy", (3, 2)),
                        ("force", (6, 6)),
                        ("virial", (3, 12)),
                        ("stress", (3, 12)),
                    ):
                        values = np.loadtxt(directory / f"{name}_train.out", ndmin=2)
                        self.assertEqual(values.shape, shape)
                        self.assertTrue(np.isfinite(values).all())
                        outputs[name] = values
                    np.testing.assert_allclose(
                        outputs["energy"][:, 1],
                        [-2 / 3, -1 / 2, 0.2],
                        rtol=2e-6,
                        err_msg=f"prediction energy order: batch {batch}, {suffix}",
                    )
                    predictions.append(outputs)
                for name in predictions[0]:
                    np.testing.assert_allclose(
                        predictions[0][name],
                        predictions[1][name],
                        rtol=3e-4,
                        atol=3e-5,
                        err_msg=f"prediction {name}: batch {batch}",
                    )


if __name__ == "__main__":
    unittest.main()

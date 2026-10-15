"""Optional native GNEP streamed-gradient regression.

Set FCVP_GNEP for the single-device case and FCVP_DEVICES=0,1 for dual GPU.
The 1 MiB run must use multiple center-atom tiles; 512 MiB fits one tile.
"""

import os
from pathlib import Path
import re
import tempfile
import unittest

import numpy as np

from test_gpu_generality import FLEXIBLE_ZBL, run_case
from reference import read_xyz


CELL = 'Lattice="100 0 0 0 100 0 0 0 100" '
PROPERTIES = 'Properties=species:S:1:pos:R:3:force:R:3\n'
CLUSTER = (
    ("Si", (0.0, 0.0, 0.0), (.1, .2, -.1)),
    ("Ge", (1.7, .2, .3), (-.2, .1, .3)),
    ("Si", (.5, 1.8, .4), (.2, -.1, .1)),
    ("Ge", (1.1, .7, 2.1), (-.1, -.2, -.3)),
)
CONFIG = (
    "type 2 Si Ge\ncutoff 5 4\nn_max 2 2\nbasis_size 3 3\n"
    "l_max 4 2 1\nneuron 8\nbatch 2 0\nepoch 1\nseed 20261015\n"
    "weight_decay 0\nlambda_e .7\nlambda_f 1.2\nlambda_v .3\n"
    "lambda_shear 1.5\ntype_weight 1.2 .8\nforce_delta .2\n"
    "zbl 2.4\n"
)
SINGLE_ATOM = (
    '1\nLattice="40 0 0 0 40 0 0 0 40" '
    'energy=0 Properties=species:S:1:pos:R:3:force:R:3\n'
    'Si 2 2 2 0 0 0\n'
)
TILE_PATTERN = re.compile(
    r"GNEP device (\d+) gradient tile atoms=(\d+) tiles=(\d+) "
    r"scratch_bytes=(\d+) untiled_bytes=([\d.eE+-]+)"
)
GRADIENT_PATTERN = re.compile(
    r"FCVP gradient check parameter (\d+) analytical ([-+.\deE]+) "
    r"finite_difference ([-+.\deE]+)"
)


def frame(cluster_count, isolated_count, with_virial):
    rows = []
    for i in range(cluster_count):
        origin = (10 + 12 * (i % 5), 10 + 12 * (i // 5), 10)
        for species, offset, force in CLUSTER:
            xyz = [origin[d] + offset[d] for d in range(3)]
            rows.append(
                f"{species} {' '.join(f'{v:.8g}' for v in xyz)} "
                f"{' '.join(f'{v:.8g}' for v in force)}\n"
            )
    for i in range(isolated_count):
        position = 90 - 10 * i
        rows.append(f"{'Si' if i == 0 else 'Ge'} {position} {position} {position} 0 0 0\n")
    n = len(rows)
    virial = (
        f'virial="{cluster_count} {cluster_count / 5} {-cluster_count / 3} '
        f'{cluster_count / 5} {2 * cluster_count} {cluster_count / 4} '
        f'{-cluster_count / 3} {cluster_count / 4} {3 * cluster_count}" '
        if with_virial else ""
    )
    return (
        f"{n}\n" + CELL + f"energy={-2 * cluster_count} "
        + f"weight={1.3 if with_virial else .7} " + virial + PROPERTIES
        + "".join(rows)
    )


FRAMES = (frame(24, 1, True), frame(25, 2, False))


def checked_gradients(stdout):
    checked = {}
    for index, analytical, finite_difference in GRADIENT_PATTERN.findall(stdout):
        checked[int(index)] = (float(analytical), float(finite_difference))
    return checked


def tile_reports(stdout):
    return [tuple(int(value) for value in match[:4]) + (float(match[4]),)
            for match in TILE_PATTERN.findall(stdout)]


def predicted_test(directory):
    return {
        name: np.loadtxt(directory / f"{name}_test.out", ndmin=2)
        for name in ("energy", "force", "virial")
    }


class GradientFixtureTests(unittest.TestCase):
    def test_unequal_structures_have_isolated_tail_atoms(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "train.xyz"
            path.write_text("".join(FRAMES))
            structures = read_xyz(path)
        self.assertEqual([len(item.species) for item in structures], [97, 102])
        self.assertIsNotNone(structures[0].virial)
        self.assertIsNone(structures[1].virial)
        for structure, isolated_count in zip(structures, (1, 2)):
            positions = structure.positions
            for index in range(len(positions) - isolated_count, len(positions)):
                others = np.delete(positions, index, axis=0)
                delta = others - positions[index]
                delta -= 100 * np.rint(delta / 100)
                self.assertGreater(np.min(np.linalg.norm(delta, axis=1)), 5)


@unittest.skipUnless(os.environ.get("FCVP_GNEP"), "Set FCVP_GNEP")
class GradientMemoryTests(unittest.TestCase):
    def run_budget(self, root, name, memory_mib, devices, mode):
        extra = {
            "projected": "fcvp 1 8 1e-2 1e-7\nlambda_2 .5\nfcvp_check 1\n",
            "checked": "fcvp_check 1\n",
            "legacy": "",
        }[mode]
        result, directory = run_case(
            root, name, CONFIG + extra + f"gradient_memory {memory_mib}\n",
            FRAMES, zbl=FLEXIBLE_ZBL, devices=devices,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        if mode != "legacy":
            self.assertIn("analytical gradient gate passed", result.stdout)
        else:
            self.assertIn("loss_rmse = 0", result.stdout)
        expected_devices = len(devices.split(","))
        reports = tile_reports(result.stdout)
        self.assertEqual(len(reports), expected_devices, result.stdout)
        for device, atoms, tiles, scratch, untiled in reports:
            self.assertGreater(atoms, 0)
            self.assertLessEqual(scratch, memory_mib * 1024 * 1024)
            if memory_mib == 1:
                self.assertGreater(tiles, 1, (device, reports))
                self.assertGreater(untiled, 1024 * 1024)
            else:
                self.assertEqual(tiles, 1, (device, reports))
                self.assertLessEqual(untiled, memory_mib * 1024 * 1024)
        gradients = checked_gradients(result.stdout)
        if mode != "legacy":
            self.assertGreaterEqual(len(gradients), 6, result.stdout)
        else:
            self.assertFalse(gradients)
        self.assertTrue(np.isfinite(np.array(list(gradients.values()))).all())
        projected = None
        if mode == "projected":
            self.assertTrue((directory / "fcvp.out").exists())
            projected = np.loadtxt(directory / "fcvp.out", ndmin=2)[:, :3]
        return {
            "gradients": gradients,
            "restart": np.loadtxt(directory / "gnep.restart"),
            "predicted": predicted_test(directory),
            "projected": projected,
        }

    def assert_budget_equivalent(self, low, high):
        self.assertEqual(low["gradients"].keys(), high["gradients"].keys())
        indices = sorted(low["gradients"])
        if indices:
            low_grad = np.array([low["gradients"][i] for i in indices])
            high_grad = np.array([high["gradients"][i] for i in indices])
            np.testing.assert_allclose(low_grad, high_grad, rtol=2e-3, atol=2e-3)
        np.testing.assert_allclose(low["restart"], high["restart"],
                                   rtol=2e-3, atol=2e-4)
        for key in ("energy", "force", "virial"):
            np.testing.assert_allclose(low["predicted"][key], high["predicted"][key],
                                       rtol=2e-3, atol=2e-4, err_msg=key)
        self.assertEqual(low["projected"] is None, high["projected"] is None)
        if low["projected"] is not None:
            np.testing.assert_allclose(low["projected"], high["projected"],
                                       rtol=2e-3, atol=2e-4)

    def test_single_gpu_streamed_gradient_matches_one_tile(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            visible = os.environ.get("FCVP_DEVICE", "0")
            low = self.run_budget(root, "single_1", 1, visible, "projected")
            high = self.run_budget(root, "single_512", 512, visible, "projected")
            self.assert_budget_equivalent(low, high)

    @unittest.skipUnless(os.environ.get("FCVP_DEVICES"), "Set FCVP_DEVICES=0,1")
    def test_dual_gpu_streamed_gradient_matches_one_tile(self):
        devices = os.environ["FCVP_DEVICES"]
        self.assertEqual(len(devices.split(",")), 2)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            low = self.run_budget(root, "dual_1", 1, devices, "checked")
            high = self.run_budget(root, "dual_512", 512, devices, "checked")
            self.assert_budget_equivalent(low, high)

    def test_legacy_loss_streamed_gradient_matches_one_tile(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            visible = os.environ.get("FCVP_DEVICE", "0")
            low = self.run_budget(root, "legacy_1", 1, visible, "legacy")
            high = self.run_budget(root, "legacy_512", 512, visible, "legacy")
            self.assert_budget_equivalent(low, high)

    def test_rejects_invalid_or_insufficient_gradient_memory(self):
        simple = (
            "type 1 Si\ncutoff 5 4\nn_max 0 0\nbasis_size 0 0\n"
            "l_max 0\nneuron 2\nbatch 1\nepoch 1\nseed 7\n"
        )
        wide = (
            "type 1 Si\ncutoff 5 4\nn_max 19 17\nbasis_size 19 17\n"
            "l_max 8 2 1\nneuron 12\nbatch 1\nepoch 1\nseed 7\n"
        )
        cases = (
            ("zero", simple + "gradient_memory 0\n", "positive integer"),
            ("negative", simple + "gradient_memory -1\n", "positive integer"),
            ("fractional", simple + "gradient_memory 1.5\n", "positive integer"),
            ("one_atom_too_wide", wide + "gradient_memory 1\n",
             "cannot hold one atom"),
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name, config, message in cases:
                with self.subTest(name=name):
                    result, _ = run_case(root, name, config, (SINGLE_ATOM,))
                    self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                    self.assertIn(message, (result.stdout + result.stderr).lower())


if __name__ == "__main__":
    unittest.main()

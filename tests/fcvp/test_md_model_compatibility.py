"""Optional GPUMD loader and exported-model E/F/V compatibility tests.

Set FCVP_GPUMD for loader checks. FCVP_GNEP and FCVP_BRIDGE also enable
exported-model comparisons with the independent NEP_CPU oracle.
"""

import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/fcvp"))
from reference import Model, Oracle, read_xyz


FRAME = (
    '4\nLattice="40 0 0 0 40 0 0 0 40" pbc="T T T" energy=-1 '
    'virial="1 .2 -.3 .2 2 .4 -.3 .4 3" '
    'Properties=species:S:1:pos:R:3:force:R:3\n'
    'Si 2 2 2 .1 .2 -.1\nGe 3.6 2.2 2.3 -.2 .1 .3\n'
    'Si 2.5 3.8 2.4 .2 -.1 .1\nGe 3.1 2.7 4.1 -.1 -.2 -.3\n'
)
DUAL_FRAME = (
    '8\nLattice="80 0 0 0 40 0 0 0 40" pbc="T T T" energy=-2 '
    'virial="2 .4 -.6 .4 4 .8 -.6 .8 6" '
    'Properties=species:S:1:pos:R:3:force:R:3\n'
    'Si 12 2 2 .1 .2 -.1\nGe 13.6 2.2 2.3 -.2 .1 .3\n'
    'Si 12.5 3.8 2.4 .2 -.1 .1\nGe 13.1 2.7 4.1 -.1 -.2 -.3\n'
    'Si 52 2 2 .1 .2 -.1\nGe 53.6 2.2 2.3 -.2 .1 .3\n'
    'Si 52.5 3.8 2.4 .2 -.1 .1\nGe 53.1 2.7 4.1 -.1 -.2 -.3\n'
)
SINGLE_TYPE_FRAME = (
    '3\nLattice="40 0 0 0 40 0 0 0 40" pbc="T T T" energy=-1 '
    'virial="1 0 0 0 1 0 0 0 1" '
    'Properties=species:S:1:pos:R:3:force:R:3\n'
    'Si 2 2 2 .1 .2 -.1\nSi 3.6 2.2 2.3 -.2 .1 .3\n'
    'Si 2.5 3.8 2.4 .1 -.3 -.2\n'
)
ELEMENTS_11 = "H He Li Be B C N O F Ne Na"


def train_model(root, name, lmax, frame=FRAME, zbl=False, extra=""):
    directory = root / name
    directory.mkdir()
    (directory / "train.xyz").write_text(frame)
    (directory / "test.xyz").write_text(frame)
    elements = "type 1 Si\n" if frame == SINGLE_TYPE_FRAME else "type 2 Si Ge\n"
    config = (
        elements + "cutoff 5 4\nn_max 2 2\nbasis_size 3 3\n"
        + f"l_max {lmax}\nneuron 8\nbatch 1\nepoch 1\nseed 137\n"
        + "weight_decay 0\nfcvp 1 10 1e-3 1e-7\nlambda_2 .1\n"
        + "lambda_e .7\nlambda_f 1.2\nlambda_v .3\n"
        + ("zbl 2.5\n" if zbl else "")
        + extra
    )
    (directory / "gnep.in").write_text(config)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("FCVP_DEVICE", "0")
    result = subprocess.run(
        [str(Path(os.environ["FCVP_GNEP"]).resolve())], cwd=directory,
        env=env, capture_output=True, text=True, timeout=240,
    )
    (directory / "stdout.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise AssertionError(result.stdout + result.stderr)
    return (directory / "nep.txt").read_text()


def model_xyz(frame):
    positions = read_xyz(frame)[0]
    lattice = " ".join(f"{value:.17g}" for value in positions.cell.ravel())
    rows = "".join(
        f"{species} {xyz[0]:.17g} {xyz[1]:.17g} {xyz[2]:.17g}\n"
        for species, xyz in zip(positions.species, positions.positions)
    )
    return (
        f"{len(positions.species)}\n"
        f'Lattice="{lattice}" pbc="T T T" '
        "Properties=species:S:1:pos:R:3\n" + rows
    )


def run_md(root, name, model, structure=None, devices=None):
    directory = root / name
    directory.mkdir()
    (directory / "nep.txt").write_text(model)
    (directory / "model.xyz").write_text(structure or (
        '2\nLattice="40 0 0 0 40 0 0 0 40" pbc="T T T" '
        'Properties=species:S:1:pos:R:3\nSi 2 2 2\nSi 3.6 2.2 2.3\n'
    ))
    (directory / "run.in").write_text(
        "potential nep.txt" + (" x\n" if devices else "\n")
        + "ensemble nve\ntime_step 0\n"
        "dump_xyz -1 0 1 snapshot.xyz potential force virial\nrun 1\n"
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = devices or os.environ.get("FCVP_DEVICE", "0")
    result = subprocess.run(
        [str(Path(os.environ["FCVP_GPUMD"]).resolve())], cwd=directory,
        env=env, capture_output=True, text=True, timeout=120,
    )
    (directory / "stdout.log").write_text(result.stdout + result.stderr)
    return result, directory


def read_md_snapshot(path):
    lines = path.read_text().splitlines()
    n = int(lines[0])
    if len(lines) != n + 2:
        raise ValueError("Expected one complete MD snapshot")
    fields = dict(token.split("=", 1) for token in shlex.split(lines[1]) if "=" in token)
    specs = fields["Properties"].split(":")
    if len(specs) % 3:
        raise ValueError("Invalid MD Properties")
    columns = {}
    width = 0
    for name, _, size in zip(specs[::3], specs[1::3], specs[2::3]):
        count = int(size)
        columns[name] = slice(width, width + count)
        width += count
    rows = [line.split() for line in lines[2:]]
    if any(len(row) != width for row in rows):
        raise ValueError("Invalid MD atom row width")
    required = {"species", "pos", "forces", "energy_atom", "virial"}
    if not required.issubset(columns):
        raise ValueError("Missing requested MD output columns")
    species = [row[columns["species"]][0] for row in rows]
    positions = np.array([row[columns["pos"]] for row in rows], float)
    forces = np.array([row[columns["forces"]] for row in rows], float)
    site_energy = np.array([row[columns["energy_atom"]][0] for row in rows], float)
    site_virial = np.array([row[columns["virial"]] for row in rows], float)
    total_energy = float(fields["energy"])
    total_virial = np.array(fields["virial"].split(), float).reshape(3, 3)
    if positions.shape != (n, 3) or forces.shape != (n, 3) or site_virial.shape != (n, 9):
        raise ValueError("Invalid MD output shape")
    np.testing.assert_allclose(site_energy.sum(), total_energy, rtol=0, atol=2e-6)
    # The header mirrors the six GPU-reduced components; the other three
    # atom columns are not inputs to that reduction.
    upper = [0, 1, 2, 4, 5, 8]
    np.testing.assert_allclose(site_virial[:, upper].sum(axis=0),
                               total_virial.ravel()[upper], rtol=0, atol=2e-6)
    np.testing.assert_allclose(total_virial, total_virial.T, rtol=0, atol=2e-6)
    return species, positions, forces, total_energy / n, total_virial.ravel()[[0, 4, 8, 1, 5, 6]] / n


def header_only(*, kind="nep5", elements="Si", nmax="2 2", basis="2 2",
                lmax="4 0 0", cutoff="5 4 4 4", zbl=None):
    names = elements.split()
    lines = [f"{kind} {len(names)} {elements}"]
    if zbl is not None:
        lines.append(f"zbl {zbl}")
    lines.extend((f"cutoff {cutoff}", f"n_max {nmax}", f"basis_size {basis}",
                  f"l_max {lmax}", "ANN 2 0"))
    return "\n".join(lines) + "\n"


class MDSnapshotReaderTests(unittest.TestCase):
    def test_header_and_atom_columns_agree_without_using_stress(self):
        snapshot = (
            '2\nLattice="40 0 0 0 40 0 0 0 40" energy=3 '
            'virial="10 10 10 10 10 10 10 10 10" '
            'stress="1 0 0 0 1 0 0 0 1" '
            'Properties=species:S:1:pos:R:3:forces:R:3:energy_atom:R:1:virial:R:9\n'
            'Si 1 2 3 .1 .2 .3 1 1 2 3 4 5 6 7 8 9\n'
            'Ge 4 5 6 -.1 -.2 -.3 2 9 8 7 0 5 4 0 0 1\n'
        )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "snapshot.xyz"
            path.write_text(snapshot)
            species, positions, force, energy, virial = read_md_snapshot(path)
        self.assertEqual(species, ["Si", "Ge"])
        np.testing.assert_array_equal(positions, [[1, 2, 3], [4, 5, 6]])
        np.testing.assert_allclose(force, [[.1, .2, .3], [-.1, -.2, -.3]])
        self.assertEqual(energy, 1.5)
        np.testing.assert_array_equal(virial, [5] * 6)


@unittest.skipUnless(os.environ.get("FCVP_GPUMD"), "Set FCVP_GPUMD")
class MDModelCompatibilityTests(unittest.TestCase):
    def assert_md_matches_oracle(self, root, name, model, frame, devices=None,
                                 oracle_model=None):
        input_path = root / (name + "_input.xyz")
        input_path.write_text(frame)
        model_path = root / (name + "_oracle.txt")
        model_path.write_text(oracle_model if oracle_model is not None else model)
        reference_frame = read_xyz(input_path)[0]
        result, directory = run_md(root, name, model, model_xyz(input_path), devices)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue((directory / "snapshot.xyz").exists())
        species, positions, forces, energy, virial = read_md_snapshot(directory / "snapshot.xyz")
        self.assertEqual(species, reference_frame.species)
        np.testing.assert_allclose(positions, reference_frame.positions,
                                   rtol=0, atol=1e-7)
        oracle = Oracle(Path(os.environ["FCVP_BRIDGE"]), model_path)
        try:
            expected = oracle.predict(reference_frame, oracle.model.beta)
        finally:
            oracle.close()
        n = len(reference_frame.species)
        self.assertGreater(np.max(np.abs(expected[1:1 + 3 * n])), 1e-3)
        self.assertGreater(np.max(np.abs(expected[-6:])), 1e-3)
        np.testing.assert_allclose(energy, expected[0], rtol=3e-3, atol=2e-4)
        np.testing.assert_allclose(forces,
                                   expected[1:1 + 3 * n].reshape(3, n).T,
                                   rtol=4e-3, atol=2e-3)
        np.testing.assert_allclose(virial, expected[-6:],
                                   rtol=4e-3, atol=2e-3)

    @unittest.skipUnless(os.environ.get("FCVP_GNEP") and os.environ.get("FCVP_BRIDGE"),
                         "Set FCVP_GNEP and FCVP_BRIDGE")
    def test_exported_nep5_three_four_five_body_efv_matches_cpu(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for lmax in ("4", "4 2 0", "4 2 1"):
                with self.subTest(lmax=lmax):
                    model = train_model(root, "train_" + lmax.replace(" ", "_"), lmax)
                    self.assert_md_matches_oracle(
                        root, "md_" + lmax.replace(" ", "_"), model, FRAME,
                    )

    @unittest.skipUnless(os.environ.get("FCVP_GNEP") and os.environ.get("FCVP_BRIDGE"),
                         "Set FCVP_GNEP and FCVP_BRIDGE")
    def test_safe_universal_zbl_model_matches_cpu(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = train_model(root, "train_zbl", "4 2 1", zbl=True)
            self.assert_md_matches_oracle(root, "md_zbl", model, FRAME)

    @unittest.skipUnless(os.environ.get("FCVP_GNEP") and os.environ.get("FCVP_BRIDGE"),
                         "Set FCVP_GNEP and FCVP_BRIDGE")
    def test_typewise_cutoff_inactive_sentinels_match_cpu(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cases = (
                ("nep_typewise", "use_typewise_cutoff 2.5 2.0\n", (2.5, 2.0, -1.0)),
                ("zbl_typewise", "use_typewise_cutoff_zbl 2.0\n", (-1.0, -1.0, 2.0)),
            )
            for name, extra, factors in cases:
                with self.subTest(name=name):
                    model = train_model(root, "train_" + name, "4 2 1",
                                        zbl=True, extra=extra)
                    lines = model.splitlines()
                    cutoff_index = next(i for i, line in enumerate(lines)
                                        if line.startswith("cutoff "))
                    cutoff = lines[cutoff_index].split()
                    self.assertEqual(len(cutoff), 8)
                    self.assertEqual(tuple(map(float, cutoff[-3:])), factors)
                    radial, angular = map(float, cutoff[1:3])
                    zbl_outer = float(next(line.split()[2] for line in lines
                                           if line.startswith("zbl ")))
                    # Si is the smallest covalent radius in this Si/Ge model.
                    # These factors cannot reduce any pair cutoff below its
                    # global cap, so the legacy CPU header is equivalent.
                    smallest_pair = 2 * 1.38667
                    for factor, cap in zip(factors, (radial, angular, zbl_outer)):
                        if factor > 0:
                            self.assertGreaterEqual(smallest_pair * factor, cap)
                    oracle_lines = lines.copy()
                    oracle_lines[cutoff_index] = " ".join(cutoff[:5])
                    oracle_model = "\n".join(oracle_lines) + "\n"
                    self.assert_md_matches_oracle(root, "md_" + name, model, FRAME,
                                                  oracle_model=oracle_model)

    @unittest.skipUnless(os.environ.get("FCVP_GNEP") and os.environ.get("FCVP_BRIDGE"),
                         "Set FCVP_GNEP and FCVP_BRIDGE")
    def test_classic_nep3_and_nep4_models_remain_accepted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            exported = train_model(root, "train_single", "4", SINGLE_TYPE_FRAME)
            source = root / "nep5_single.txt"
            source.write_text(exported)
            model = Model.read(source)
            self.assertEqual(len(model.elements), 1)
            shared = model.shared_bias_index
            self.assertIsNotNone(shared)
            # Amplify the synthetic inference fixture so force and virial
            # comparisons cannot pass on a nearly constant potential.
            model.values[model.indices] *= 100.0
            model.values[shared] *= 100.0
            for version in ("nep3", "nep4"):
                with self.subTest(version=version):
                    values = np.delete(model.values, shared - 1)
                    values[shared - 1] += model.values[shared - 1]
                    lines = model.header.copy()
                    lines[0] = lines[0].replace("nep5", version, 1)
                    legacy = "\n".join(lines) + "\n" + "\n".join(
                        f"{value:.17e}" for value in values
                    ) + "\n"
                    self.assert_md_matches_oracle(root, "md_" + version,
                                                  legacy, SINGLE_TYPE_FRAME)

    def test_loader_rejects_models_beyond_fixed_array_capacities(self):
        cases = (
            ("basis", header_only(basis="19 2"), "basis capacity"),
            ("angular_moments", header_only(nmax="2 17"), "moment capacity"),
            ("angular_dimension", header_only(nmax="0 10", lmax="8 2 1"),
             "descriptor dimension"),
            ("total_dimension", header_only(nmax="19 9", lmax="8 2 0"),
             "descriptor dimension"),
            ("flexible_11", header_only(kind="nep5_zbl", elements=ELEMENTS_11,
                                        zbl="0 0"), "coefficient capacity"),
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name, model, message in cases:
                structure = None
                if name == "flexible_11":
                    structure = (
                        '2\nLattice="40 0 0 0 40 0 0 0 40" pbc="T T T" '
                        'Properties=species:S:1:pos:R:3\nH 2 2 2\nHe 3.6 2.2 2.3\n'
                    )
                devices_list = [None]
                if os.environ.get("FCVP_DEVICES"):
                    devices_list.append(os.environ["FCVP_DEVICES"])
                for devices in devices_list:
                    with self.subTest(name=name, devices=devices):
                        suffix = "dual" if devices else "single"
                        result, directory = run_md(root, name + "_" + suffix,
                                                   model, structure, devices)
                        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                        self.assertIn(message, (result.stdout + result.stderr).lower())
                        self.assertFalse((directory / "snapshot.xyz").exists())

    @unittest.skipUnless(os.environ.get("FCVP_GNEP") and os.environ.get("FCVP_BRIDGE")
                         and os.environ.get("FCVP_DEVICES"),
                         "Set FCVP_GNEP, FCVP_BRIDGE and FCVP_DEVICES=0,1")
    def test_exported_five_body_dual_gpu_efv_matches_cpu(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = train_model(root, "train_dual", "4 2 1")
            self.assert_md_matches_oracle(root, "md_dual", model, DUAL_FRAME,
                                          os.environ["FCVP_DEVICES"])

    @unittest.skipUnless(os.environ.get("FCVP_GNEP"), "Set FCVP_GNEP")
    def test_loader_rejects_zbl_past_angular_neighbor_coverage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            exported = train_model(root, "train_plain", "4")
            lines = exported.splitlines()
            lines[0] = lines[0].replace("nep5 ", "nep5_zbl ", 1)
            lines.insert(1, "zbl 2.25 4.5")
            result, directory = run_md(root, "zbl_cutoff", "\n".join(lines) + "\n",
                                       model_xyz_from_text(FRAME))
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn("angular neighbor coverage", (result.stdout + result.stderr).lower())
            self.assertFalse((directory / "snapshot.xyz").exists())


def model_xyz_from_text(frame):
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "frame.xyz"
        path.write_text(frame)
        return model_xyz(path)


if __name__ == "__main__":
    unittest.main()

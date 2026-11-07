"""Optional native regression: FCVP_GNEP=/path/to/gnep python -m unittest discover -s tests/fcvp -p test_gpu_virial.py -v."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np


@unittest.skipUnless(
    os.environ.get("FCVP_GNEP"), "Set FCVP_GNEP to run the native GPU test"
)
class VirialReportingTests(unittest.TestCase):
    def test_validation_virial_is_independent_of_shear_weight(self):
        binary = Path(os.environ["FCVP_GNEP"]).resolve()
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = os.environ.get("FCVP_DEVICE", "0")
        frame = (
            "2\n"
            'Lattice="30 0 0 0 30 0 0 0 30" energy=0 '
            'virial="0 20 40 20 0 60 40 60 0" '
            "Properties=species:S:1:pos:R:3:force:R:3\n"
            "Si 1 1 1 0 0 0\n"
            "Si 3.3 2.2 1.7 0 0 0\n"
        )
        common = (
            "type 1 Si\ncutoff 5 4\nn_max 4 4\nbasis_size 8 8\n"
            "neuron 30\nl_max 4\nbatch 1\nepoch 1\nseed 271828\n"
            "loss_rmse 1e-8\nweight_decay 0\n"
            "lambda_e 0\nlambda_f 0\nlambda_v 0\n"
        )
        train_rmse, validation_rmse = [], []
        with tempfile.TemporaryDirectory() as temporary:
            for shear in (0, 2):
                directory = Path(temporary) / str(shear)
                directory.mkdir()
                for name in ("train.xyz", "test.xyz"):
                    (directory / name).write_text(frame)
                config = common + f"lambda_shear {shear}\n"
                (directory / "gnep.in").write_text(config)
                result = subprocess.run(
                    [str(binary)],
                    cwd=directory,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                row = np.loadtxt(directory / "loss.out", ndmin=2)[-1]
                train_rmse.append(row[4])
                validation_rmse.append(row[7])

                # With zero loss weights, the exported model still has the
                # initialization. Its explicit tensor residuals are an
                # independent reference for the reported validation metric.
                (directory / "gnep.in").write_text(config + "prediction 1\n")
                result = subprocess.run(
                    [str(binary)],
                    cwd=directory,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                virial = np.loadtxt(directory / "virial_train.out", ndmin=2)
                raw_rmse = np.sqrt(np.mean((virial[:, :6] - virial[:, 6:]) ** 2))
                self.assertAlmostEqual(row[7], raw_rmse, delta=2e-4)

            self.assertGreater(abs(train_rmse[0] - train_rmse[1]), 1.0)
            self.assertAlmostEqual(validation_rmse[0], validation_rmse[1], delta=2e-5)


if __name__ == "__main__":
    unittest.main()

"""Exercise timing/checkpoint process control without claiming GPU execution."""

import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/fcvp"))
from benchmark_gpu import gnep_config, run


class BenchmarkConfigTests(unittest.TestCase):
    def test_unpatched_gnep_receives_no_new_keywords(self):
        common = "type 2 Pb Te\nbatch 32\n"
        legacy = gnep_config(common, 200, 1234, "", upstream=True)
        patched = gnep_config(common, 200, 1234, "")
        self.assertNotIn("seed ", legacy)
        self.assertNotIn("loss_rmse", legacy)
        self.assertNotIn("fcvp", legacy)
        self.assertEqual(patched, legacy + "seed 1234\n")

    def test_fcvp_controls_keep_seed_and_projection_options(self):
        config = gnep_config("type 2 Pb Te\n", 200, 1234, "fcvp 5 40 1e-8 1e-8\n")
        self.assertIn("seed 1234\n", config)
        self.assertIn("fcvp 5 40 1e-8 1e-8\n", config)


@unittest.skipUnless(
    os.name == "posix", "GPU benchmark process control uses POSIX signals"
)
class BenchmarkProcessTests(unittest.TestCase):
    def invoke(self, body, timeout=5, snes=False):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            binary = directory / "fake_trainer"
            binary.write_text(f"#!{sys.executable}\n" + body)
            binary.chmod(0o755)
            result = run(
                binary, directory, os.environ.copy(), [0.001, 0.05, None], timeout, snes
            )
            checkpoint = directory / "nep_at_target.txt"
            return result, checkpoint.read_text() if checkpoint.exists() else None

    def test_gnep_report_preserves_checkpoint_and_wall_time(self):
        result, checkpoint = self.invoke(
            "from pathlib import Path\nimport time\n"
            "Path('nep.txt').write_text('checkpoint at epoch 1')\n"
            "time.sleep(.05)\n"
            "print('1 .1 .01 .04 0 .0005 .04 0 .001 999',flush=True)\n"
            "time.sleep(30)\n"
        )
        self.assertEqual(result["status"], "target_reached")
        self.assertEqual(checkpoint, "checkpoint at epoch 1")
        self.assertGreater(result["time_to_target_seconds"], 0.05)
        self.assertLess(result["time_to_target_seconds"], 5)

    def test_snes_report_below_training_target_does_not_pass_validation(self):
        result, checkpoint = self.invoke(
            "print('100 .1 0 0 .0001 .001 0 .01 .1 0',flush=True)\n", snes=True
        )
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertIsNone(result["time_to_target_seconds"])
        self.assertIsNone(checkpoint)

    def test_timeout_stops_process_without_reporting_success(self):
        result, checkpoint = self.invoke(
            "import signal,time\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)\ntime.sleep(30)\n",
            timeout=0.2,
        )
        self.assertTrue(result["timed_out"])
        self.assertIsNone(result["time_to_target_seconds"])
        self.assertIsNone(checkpoint)
        self.assertLess(result["process_wall_seconds"], 5)


if __name__ == "__main__":
    unittest.main()

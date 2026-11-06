"""Optional native LM input tests.

Set NEP_LM_GNEP to a built gnep and NEP_LM_DEVICE to a free GPU (default 0).
NEP_LM_BASELINE enables legacy binary parity; NEP_LM_DEVICES=0,1 enables
the two-visible-device rejection check, which must fail before data allocation.
The synthetic fixtures test software behavior, not optimizer accuracy.
"""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np


FRAME_A = (
    '2\nLattice="30 0 0 0 30 0 0 0 30" energy=-1 weight=1.2 '
    'virial=".1 .02 .03 .02 .2 .04 .03 .04 .3" '
    'Properties=species:S:1:pos:R:3:force:R:3\n'
    'Si 2 2 2 .03 -.02 .01\nGe 3.8 2.1 2.2 -.03 .02 -.01\n'
)
FRAME_B = (
    '3\nLattice="30 0 0 0 30 0 0 0 30" energy=-1.5 weight=.8 '
    'Properties=species:S:1:pos:R:3:force:R:3\n'
    'Ge 2 2 2 .02 .01 -.01\nSi 3.9 2.2 2.3 -.01 -.02 .01\n'
    'Si 2.3 4 2.4 -.01 .01 0\n'
)
FRAMES = (FRAME_A, FRAME_B)
GUARD = (
    FRAME_A.replace('energy=-1 ', 'energy=-1.02 ').replace('Ge 3.8', 'Ge 3.85'),
)
CHANGED_TEST = (
    FRAME_B.replace('energy=-1.5 ', 'energy=37 ').replace('Ge 2 2 2', 'Ge 7 7 7'),
)
COMMON = (
    'type 2 Si Ge\ncutoff 5 4\nn_max 0 0\nbasis_size 0 0\n'
    'l_max 1 0 0\nneuron 2\nbatch 2 0\nepoch 2\nseed 20261015\n'
    'weight_decay 0\nlambda_e 1\nlambda_f 2\nlambda_v .1\n'
    'lambda_1 0\nlambda_2 0\n'
)
# The upstream baseline accepts neither 4/5-body l_max fields nor lambda_1/2.
BASELINE_COMMON = COMMON.replace('l_max 1 0 0', 'l_max 1').replace('lambda_1 0\nlambda_2 0\n', '')
LM = 'optimizer lm\nloss_rmse 1e-7\nlm 2 4 .2 .1\nlm_guard guard.xyz 1.1\n'


def dense_guard():
    rows = []
    for i in range(8):
        x, y, z = (2 + 1.1 * ((i >> d) & 1) for d in range(3))
        species = 'Si' if i % 2 == 0 else 'Ge'
        rows.append(f'{species} {x:.8g} {y:.8g} {z:.8g} .01 -.01 0\n')
    frame = (
        '8\nLattice="30 0 0 0 30 0 0 0 30" energy=-4 '
        'virial=".2 0 0 0 .2 0 0 0 .2" '
        'Properties=species:S:1:pos:R:3:force:R:3\n' + ''.join(rows)
    )
    return (frame, frame.replace('energy=-4 ', 'energy=-3.9 '),
            frame.replace('energy=-4 ', 'energy=-4.1 '))


@unittest.skipUnless(os.environ.get('NEP_LM_GNEP'), 'Set NEP_LM_GNEP for native GPU tests')
class InputAndCompatibilityTests(unittest.TestCase):
    def run_case(self, root, name, extra='', frames=FRAMES, guard=GUARD,
                 test_frames=FRAMES, binary=None, devices=None, model=None, setup=None,
                 common=COMMON):
        folder = root / name
        folder.mkdir()
        (folder / 'train.xyz').write_text(''.join(frames))
        (folder / 'test.xyz').write_text(''.join(test_frames))
        if guard is not None:
            (folder / 'guard.xyz').write_text(''.join(guard))
        (folder / 'gnep.in').write_text(common + extra)
        if model is not None:
            (folder / 'nep.txt').write_bytes(model)
        if setup is not None:
            setup(folder)
        environment = os.environ.copy()
        for key in tuple(environment):
            if key.startswith('GPUMD_LM_') or key.startswith('GPUMD_EXPERIMENT_'):
                environment.pop(key)
        environment['CUDA_VISIBLE_DEVICES'] = devices or os.environ.get('NEP_LM_DEVICE', '0')
        result = subprocess.run(
            [str(Path(binary or os.environ['NEP_LM_GNEP']).resolve())],
            cwd=folder, env=environment, capture_output=True, text=True, timeout=180,
        )
        (folder / 'stdout.log').write_text(result.stdout + result.stderr)
        return result, folder

    def assert_success(self, result, folder):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        values = np.loadtxt(folder / 'gnep.restart', ndmin=1)
        self.assertEqual(values.size, 26)
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue((folder / 'nep.txt').is_file())

    def assert_failure(self, result, message):
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(message, result.stdout + result.stderr)

    def test_native_lm_reads_dedicated_guard_and_gradient_check(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, folder = self.run_case(Path(temporary), 'lm', LM + 'lm_check 1\n')
            self.assert_success(result, folder)
            self.assertIn('Constructing LM guard_set', result.stdout)
            self.assertIn('LM native gradient gate', result.stdout)
            table = np.loadtxt(folder / 'lm.out', ndmin=2)
            self.assertEqual(table.shape[0], 2)
            self.assertTrue(np.isfinite(table).all())
            self.assertTrue(np.any(table[:, 3] == 1), table)
            self.assertTrue(np.all(table[:, 2] <= table[:, 1] + 1e-8), table)

    def test_fast_path_passes_gradient_check_and_lowers_objective(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, folder = self.run_case(Path(temporary), 'lm', LM + 'lm_fast 1\nlm_check 1\n')
            self.assert_success(result, folder)
            self.assertIn('LM fast path', result.stdout)
            self.assertIn('LM native gradient gate', result.stdout)
            table = np.loadtxt(folder / 'lm.out', ndmin=2)
            self.assertTrue(np.isfinite(table).all())
            self.assertTrue(np.any(table[:, 3] == 1), table)
            self.assertTrue(np.all(table[:, 2] <= table[:, 1] + 1e-8), table)

    def test_published_example_uses_lm_defaults(self):
        root = Path(__file__).resolve().parents[2]
        template = (root / 'examples/nep_train/lm/gnep.in').read_text()

        def setup(folder):
            (folder / 'gnep.in').write_text(template)

        with tempfile.TemporaryDirectory() as temporary:
            result, folder = self.run_case(Path(temporary), 'example', setup=setup)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('probes=16 candidates=8', result.stdout)
            parameters = np.loadtxt(folder / 'gnep.restart', ndmin=1)
            self.assertEqual(parameters.size, 4026)
            self.assertTrue(np.isfinite(parameters).all())
            rows = np.loadtxt(folder / 'lm.out', ndmin=2)
            self.assertEqual(rows.shape, (10, 10))
            self.assertTrue(np.isfinite(rows).all())

    def test_rejects_invalid_optimizer_and_lm_values(self):
        cases = (
            ('optimizer\n', 'optimizer requires'),
            ('optimizer unknown\n', 'optimizer requires'),
            ('optimizer lm extra\n', 'optimizer requires'),
            ('optimizer lm\nlm 2 4 .2\n', 'lm requires'),
            ('optimizer lm\nlm 0 4 .2 .1\n', 'lm requires'),
            ('optimizer lm\nlm 1025 4 .2 .1\n', 'lm requires'),
            ('optimizer lm\nlm 1.5 4 .2 .1\n', 'lm requires'),
            ('optimizer lm\nlm 2 0 .2 .1\n', 'lm requires'),
            ('optimizer lm\nlm 2 65 .2 .1\n', 'lm requires'),
            ('optimizer lm\nlm 2 4 0 .1\n', 'lm requires'),
            ('optimizer lm\nlm 2 4 1.1 .1\n', 'lm requires'),
            ('optimizer lm\nlm 2 4 nan .1\n', 'lm requires'),
            ('optimizer lm\nlm 2 4 .2 0\n', 'lm requires'),
            ('optimizer lm\nlm 2 4 .2 1e-10\n', 'lm requires'),
            ('optimizer lm\nlm 2 4 .2 1e10\n', 'lm requires'),
            ('optimizer lm\nlm 2 4 .2 inf\n', 'lm requires'),
            ('optimizer lm\nlm_guard guard.xyz\n', 'lm_guard requires'),
            ('optimizer lm\nlm_guard guard.xyz .99\n', 'lm_guard requires'),
            ('optimizer lm\nlm_guard guard.xyz nan\n', 'lm_guard requires'),
            ('optimizer lm\nlm_check 2\n', 'lm_check requires'),
            ('optimizer lm\nlm_check 0.5\n', 'lm_check requires'),
            ('optimizer lm\nlm_fast 2\n', 'lm_fast requires'),
            ('optimizer lm\nlm_fast 0.5\n', 'lm_fast requires'),
            ('lm_fast 1\n', 'lm_fast require optimizer lm'),
        )
        with tempfile.TemporaryDirectory() as temporary:
            for index, (extra, message) in enumerate(cases):
                with self.subTest(extra=extra):
                    result, _ = self.run_case(Path(temporary), str(index), extra)
                    self.assert_failure(result, message)

    def test_rejects_incompatible_training_options(self):
        cases = (
            ('optimizer lm\nloss_rmse 1e-7\n', 'explicit nonempty lm_guard'),
            (LM.replace('loss_rmse 1e-7\n', ''), 'requires loss_rmse'),
            (LM + 'lambda_1 .1\n', 'requires loss_rmse'),
            (LM + 'weight_decay .01\n', 'requires loss_rmse'),
            (LM + 'batch 1 1\n', 'requires loss_rmse'),
            (LM + 'prediction 1\n', 'omit it for prediction'),
            ('lm 2 4 .2 .1\n', 'require optimizer lm'),
            ('lm_guard guard.xyz 1.1\n', 'require optimizer lm'),
            ('lm_check 0\n', 'require optimizer lm'),
        )
        with tempfile.TemporaryDirectory() as temporary:
            for index, (extra, message) in enumerate(cases):
                with self.subTest(extra=extra):
                    result, _ = self.run_case(Path(temporary), str(index), extra)
                    self.assert_failure(result, message)

    def test_missing_or_empty_guard_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            for name, guard in (('missing', None), ('empty', ())):
                with self.subTest(name=name):
                    result, folder = self.run_case(Path(temporary), name, LM, guard=guard)
                    self.assert_failure(result, 'Failed to open guard.xyz' if guard is None
                                        else 'contains no structures')
                    self.assertFalse((folder / 'lm.out').exists())

    def test_rejects_reserved_guard_files_and_path_aliases(self):
        with tempfile.TemporaryDirectory() as temporary:
            for index, filename in enumerate(('train.xyz', 'test.xyz', './train.xyz', './test.xyz')):
                with self.subTest(filename=filename):
                    extra = LM.replace('lm_guard guard.xyz', 'lm_guard ' + filename)
                    result, folder = self.run_case(Path(temporary), str(index), extra)
                    self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertTrue('dedicated file' in result.stdout + result.stderr or
                                    'must not alias' in result.stdout + result.stderr)
                    self.assertFalse((folder / 'lm.out').exists())

    def test_default_and_explicit_adam_are_identical(self):
        with tempfile.TemporaryDirectory() as temporary:
            outputs = []
            for name, extra in (('default', ''), ('adam', 'optimizer adam\n')):
                result, folder = self.run_case(Path(temporary), name, extra)
                self.assert_success(result, folder)
                self.assertFalse((folder / 'lm.out').exists())
                outputs.append({name: (folder / name).read_bytes()
                                for name in ('nep.txt', 'gnep.restart')})
            self.assertEqual(outputs[0], outputs[1])

    def test_guard_cannot_alias_native_outputs_and_input_is_preserved(self):
        guard_bytes = ''.join(GUARD).encode()
        cases = (
            ('lm.out', 'direct'),
            ('nep.txt', 'direct'),
            ('force_test.out', 'direct'),
            ('lm.out', 'dot_path'),
            ('nep.txt', 'input_symlink'),
            ('force_test.out', 'input_hardlink'),
            ('lm.out', 'output_symlink'),
            ('nep.txt', 'output_hardlink'),
        )
        with tempfile.TemporaryDirectory() as temporary:
            for index, (output_name, kind) in enumerate(cases):
                with self.subTest(output=output_name, kind=kind):
                    guard_name = ('./' + output_name if kind == 'dot_path' else
                                  output_name if kind == 'direct' else 'guard.xyz')

                    def setup(folder):
                        output_path = folder / output_name
                        guard_path = folder / 'guard.xyz'
                        if kind in ('direct', 'dot_path'):
                            output_path.write_bytes(guard_bytes)
                        elif kind.startswith('input_'):
                            guard_path.unlink()
                            output_path.write_bytes(guard_bytes)
                            if kind == 'input_symlink':
                                guard_path.symlink_to(output_name)
                            else:
                                os.link(output_path, guard_path)
                        elif kind == 'output_symlink':
                            output_path.symlink_to('guard.xyz')
                        else:
                            os.link(guard_path, output_path)

                    extra = LM.replace('lm_guard guard.xyz', 'lm_guard ' + guard_name)
                    result, folder = self.run_case(Path(temporary), str(index), extra, setup=setup)
                    self.assert_failure(result, 'must not alias')
                    self.assertEqual((folder / output_name).read_bytes(), guard_bytes)
                    self.assertEqual((folder / 'guard.xyz').read_bytes(), guard_bytes)
                    self.assertFalse((folder / 'loss.out').exists())
                    self.assertFalse((folder / 'gnep.restart').exists())
                    self.assertNotIn('LM parameters=', result.stdout)

            result, folder = self.run_case(
                Path(temporary), 'missing_reserved',
                LM.replace('lm_guard guard.xyz', 'lm_guard ./lm.out'), guard=None)
            self.assert_failure(result, 'must not alias')
            self.assertFalse((folder / 'lm.out').exists())

    def test_guard_with_output_basename_in_another_directory_is_valid(self):
        guard_bytes = ''.join(GUARD).encode()

        def setup(folder):
            nested = folder / 'nested'
            nested.mkdir()
            (nested / 'lm.out').write_bytes(guard_bytes)

        with tempfile.TemporaryDirectory() as temporary:
            result, folder = self.run_case(
                Path(temporary), 'nested_guard',
                LM.replace('lm_guard guard.xyz', 'lm_guard nested/lm.out'), setup=setup)
            self.assert_success(result, folder)
            self.assertEqual((folder / 'nested' / 'lm.out').read_bytes(), guard_bytes)
            self.assertTrue((folder / 'lm.out').is_file())

    @unittest.skipUnless(os.environ.get('NEP_LM_BASELINE'), 'Set NEP_LM_BASELINE')
    def test_default_matches_baseline_binary(self):
        with tempfile.TemporaryDirectory() as temporary:
            outputs = []
            for name, binary in (('baseline', os.environ['NEP_LM_BASELINE']),
                                 ('native', os.environ['NEP_LM_GNEP'])):
                result, folder = self.run_case(
                    Path(temporary), name, binary=binary, common=BASELINE_COMMON)
                self.assert_success(result, folder)
                outputs.append({name: (folder / name).read_bytes()
                                for name in ('nep.txt', 'gnep.restart')})
            self.assertEqual(outputs[0], outputs[1])

    def test_changing_test_file_does_not_change_lm_acceptance(self):
        with tempfile.TemporaryDirectory() as temporary:
            outputs = []
            traces = []
            for name, test_frames in (('original', FRAMES), ('changed', CHANGED_TEST)):
                result, folder = self.run_case(Path(temporary), name, LM, test_frames=test_frames)
                self.assert_success(result, folder)
                outputs.append((folder / 'gnep.restart').read_bytes())
                trace = np.loadtxt(folder / 'lm.out', ndmin=2)
                self.assertTrue(np.any(trace[:, 3] == 1), trace)
                traces.append(trace[:, [0, 1, 2, 3, 4, 7, 8, 9]])
            self.assertEqual(outputs[0], outputs[1])
            np.testing.assert_array_equal(traces[0], traces[1])

    def test_guard_larger_and_denser_than_training_fits_native_capacity(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, folder = self.run_case(Path(temporary), 'dense_guard', LM + 'epoch 1\n',
                                           guard=dense_guard())
            self.assert_success(result, folder)
            self.assertIn('Constructing LM guard_set', result.stdout)
            self.assertTrue(np.isfinite(np.loadtxt(folder / 'lm.out', ndmin=2)).all())

    @unittest.skipUnless(os.environ.get('NEP_LM_DEVICES'), 'Set NEP_LM_DEVICES=0,1')
    def test_two_visible_gpus_fail_before_data_allocation(self):
        devices = os.environ['NEP_LM_DEVICES']
        self.assertGreaterEqual(len(devices.split(',')), 2)
        with tempfile.TemporaryDirectory() as temporary:
            result, folder = self.run_case(Path(temporary), 'two_devices', LM, devices=devices)
            self.assert_failure(result, 'exactly one visible GPU')
            self.assertNotIn('Constructing train_set', result.stdout)
            self.assertNotIn('Constructing LM guard_set', result.stdout)
            self.assertFalse((folder / 'lm.out').exists())

    def test_prediction_uses_model_without_lm_options_or_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trained, folder = self.run_case(root, 'trained', 'epoch 1\n')
            self.assert_success(trained, folder)
            model = (folder / 'nep.txt').read_bytes()
            result, predicted = self.run_case(root, 'predicted', 'prediction 1\n',
                                               guard=None, model=model)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual((predicted / 'nep.txt').read_bytes(), model)
            self.assertFalse((predicted / 'lm.out').exists())
            for name in ('energy', 'force', 'virial'):
                values = np.loadtxt(predicted / (name + '_train.out'), ndmin=2)
                self.assertTrue(np.isfinite(values).all(), name)


if __name__ == '__main__':
    unittest.main()

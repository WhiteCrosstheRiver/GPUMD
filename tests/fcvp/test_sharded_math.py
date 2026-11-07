"""Global sufficient statistics must not depend on the number of shards."""

import unittest

import numpy as np


class ShardedMathTests(unittest.TestCase):
    def test_uneven_shards_preserve_objective_gradient_and_gram(self):
        rng = np.random.default_rng(42)
        a = rng.normal(size=(37, 9))
        y = rng.normal(size=37)
        beta = rng.normal(size=9)
        epsilon = 1e-4
        residual = a @ beta - y
        mse = residual @ residual / len(y)
        gradient = a.T @ residual / (len(y) * np.sqrt(mse + epsilon**2))
        for devices in (1, 2, 3, 4, 8, 16):
            partitions = np.array_split(np.arange(len(y)), devices)
            h = sum(a[i].T @ a[i] for i in partitions) / len(y)
            g = sum(a[i].T @ y[i] for i in partitions) / len(y)
            yy = sum(y[i] @ y[i] for i in partitions) / len(y)
            np.testing.assert_allclose(h, a.T @ a / len(y), atol=1e-14)
            np.testing.assert_allclose(g, a.T @ y / len(y), atol=1e-14)
            self.assertAlmostEqual(beta @ h @ beta - 2 * g @ beta + yy, mse)
            global_mse = sum(residual[i] @ residual[i] for i in partitions) / len(y)
            reduced_gradient = sum(
                a[i].T @ residual[i] / (len(y) * np.sqrt(global_mse + epsilon**2))
                for i in partitions
            )
            np.testing.assert_allclose(reduced_gradient, gradient, atol=1e-14)

    def test_averaging_local_rmse_is_not_global_rmse(self):
        residuals = (np.array([1.0]), np.array([3.0, 3.0, 3.0]))
        global_rmse = np.sqrt(sum(r @ r for r in residuals) / 4)
        wrong = np.mean([np.sqrt(np.mean(r**2)) for r in residuals])
        self.assertGreater(abs(global_rmse - wrong), 0.5)


if __name__ == "__main__":
    unittest.main()

import unittest
import numpy as np
from plot import relative_error, discarded_errors


class SvdErrors(unittest.TestCase):
    def test_errors_use_frobenius_norm_and_discarded_tail(self):
        matrix = np.diag([4.0, 3.0])
        rank_one = np.diag([4.0, 0.0])
        self.assertAlmostEqual(relative_error(matrix, rank_one), 0.6)
        np.testing.assert_allclose(discarded_errors(np.array([4.0, 3.0])), [1.0, 0.6, 0.0])


if __name__ == '__main__':
    unittest.main()

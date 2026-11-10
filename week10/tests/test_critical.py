import unittest
import numpy as np
from critical import specific_heat, parabola_peak


class CriticalAnalysis(unittest.TestCase):
    def test_second_derivative_and_parabolic_peak(self):
        k = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
        np.testing.assert_allclose(specific_heat(k, 3 * k**2 + 2*k + 1), 6*k[1:-1]**2)
        t = np.array([1.0, 2.0, 3.0])
        x, y = parabola_peak(t, 5-(t-2.2)**2)
        self.assertAlmostEqual(x, 2.2)
        self.assertAlmostEqual(y, 5.0)

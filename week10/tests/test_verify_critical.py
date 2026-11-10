import unittest
import numpy as np
from verify_critical import enumerate_ising


class IsingEnumeration(unittest.TestCase):
    def test_two_by_two_zero_coupling(self):
        g, cv = enumerate_ising(2, 0.0)
        self.assertAlmostEqual(g, np.log(2.0))
        self.assertEqual(cv, 0.0)

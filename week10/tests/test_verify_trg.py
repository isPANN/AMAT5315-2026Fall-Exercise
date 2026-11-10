import unittest
import numpy as np
from verify_trg import enumerate_periodic


class PeriodicEnumeration(unittest.TestCase):
    def test_two_by_two_parallel_bonds(self):
        k = 0.44
        g, _ = enumerate_periodic(2, k)
        self.assertAlmostEqual(g, np.log(4*np.cosh(8*k)+12)/4)

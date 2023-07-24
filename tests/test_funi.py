import unittest

import numpy as np

import funi  # isort:skip


class FuniTest(unittest.TestCase):
    def setUp(self):
        # stack 100 x 3 arrays 3 times then shuffle
        q = np.random.rand(100, 3)
        self.q = np.vstack((q, q, q))
        np.random.shuffle(self.q)

    def test_all_return(self):
        tol = 1e-10

        # sorted indices
        unique_s, ids_s, inv_s = funi.unique_rows(self.q, tol, True, "l")

        # mapping check
        assert np.allclose(self.q[ids_s], unique_s)
        assert np.allclose(self.q, unique_s[inv_s])

        # sortedness check
        assert np.all(np.diff(ids_s) > 0)

        # not sorted
        # sorted indices
        unique_ns, ids_ns, inv_ns = funi.unique_rows(self.q, tol, False, "l")

        # mapping check
        assert np.allclose(self.q[ids_ns], unique_ns)
        assert np.allclose(self.q, unique_ns[inv_ns])

        # now for axis based
        # sorted indices
        unique_s, ids_s, inv_s = funi.unique_rows(self.q, tol, True, "a")

        # mapping check
        assert np.allclose(self.q[ids_s], unique_s)
        assert np.allclose(self.q, unique_s[inv_s])

        # sortedness check
        assert np.all(np.diff(ids_s) > 0)

        # not sorted
        # sorted indices
        unique_ns, ids_ns, inv_ns = funi.unique_rows(self.q, tol, False, "a")

        # mapping check
        assert np.allclose(self.q[ids_ns], unique_ns)
        assert np.allclose(self.q, unique_ns[inv_ns])

    def test_cross_compare(self):
        # Compare the results against each other
        tol = 1e-10
        a_unique_s, a_ids_s, a_inv_s = funi.unique_rows(self.q, tol, True, "a")
        l_unique_s, l_ids_s, l_inv_s = funi.unique_rows(self.q, tol, True, "l")

        assert np.allclose(a_unique_s, l_unique_s)
        assert np.all(a_ids_s == l_ids_s)
        assert np.all(a_inv_s == l_inv_s)


if __name__ == "__main__":
    unittest.main()

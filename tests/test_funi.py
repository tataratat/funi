import unittest

import numpy as np

import funi


class FuniTest(unittest.TestCase):
    def setUp(self):
        # stack 100 x 3 arrays 3 times then shuffle
        q = np.random.rand(100, 3)
        self.q = np.vstack((q, q, q))
        np.random.shuffle(self.q)

    def test_all_return(self):
        tol = 1e-10

        # sorted indices
        unique_s, ids_s, inv_s = funi.unique_float64(
            self.q, tol, True, True, True, True
        )

        # mapping check
        assert np.allclose(self.q[ids_s], unique_s)
        assert np.allclose(self.q, unique_s[inv_s])

        # sortedness check
        assert ~np.all(np.diff(ids_s) < 0)

        # not sorted
        # sorted indices
        unique_ns, ids_ns, inv_ns = funi.unique_float64(
            self.q, tol, True, True, False, True
        )

        # mapping check
        assert np.allclose(self.q[ids_ns], unique_ns)
        assert np.allclose(self.q, unique_ns[inv_ns])

        # no unique return - sorted
        unique, ids, inv = funi.unique_float64(
            self.q, tol, False, True, True, True
        )

        assert unique.size == 0
        assert np.allclose(self.q, self.q[ids][inv])
        assert np.allclose(ids, ids_s)
        assert np.allclose(inv, inv_s)

        # no unique return - not sorted
        unique, ids, inv = funi.unique_float64(
            self.q, tol, False, True, False, True
        )

        assert unique.size == 0
        assert np.allclose(self.q, self.q[ids][inv])
        assert np.allclose(ids, ids_ns)
        assert np.allclose(inv, inv_ns)

        # no inverse - sorted
        unique, ids, inv = funi.unique_float64(
            self.q, tol, True, True, True, False
        )

        assert inv.size == 0
        assert np.allclose(unique, self.q[ids])
        assert np.allclose(ids, ids_s)

        # no invese - not sorted
        unique, ids, inv = funi.unique_float64(
            self.q, tol, True, True, False, False
        )

        assert inv.size == 0
        assert np.allclose(unique, self.q[ids])
        assert np.allclose(ids, ids_ns)


if __name__ == "__main__":
    unittest.main()

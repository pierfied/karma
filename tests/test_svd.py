import numpy as np
from karma.tools import svd


def test_svd_eye():
    """Test that the SVD of the identity matrix is the identity matrix."""

    n = 10

    mat = np.eye(n)

    U, S, VT = svd.svd(mat)

    np.testing.assert_almost_equal(U, np.eye(n))
    np.testing.assert_almost_equal(S, 1)
    np.testing.assert_almost_equal(VT, np.eye(n))


def test_svd_diag():
    """Test that the SVD of a sorted diagonal matrix is."""

    n = 10

    diag_elem = np.flip(np.sort(np.random.uniform(size=n)))
    mat = np.diag(diag_elem)

    U, S, VT = svd.svd(mat)

    np.testing.assert_almost_equal(U, np.eye(n))
    np.testing.assert_almost_equal(S, diag_elem)
    np.testing.assert_almost_equal(VT, np.eye(n))


def test_svd_rand():
    """Test that the SVD of a random matrix recovers the original matrix."""

    n = 10

    mat = np.random.uniform(size=(n, n))

    U, S, VT = svd.svd(mat)

    np.testing.assert_almost_equal(U @ np.diag(S) @ VT, mat)


def test_svd_dim_err():
    """Test that svd raises an exception when a non-2D array is passed."""

    n = 10

    mat = np.zeros((n, n, n))

    with np.testing.assert_raises_regex(Exception, '2D'):
        svd.svd(mat)


def test_svd_non_sqr_err():
    """Test that svd raises an exception when a non-square matrix is passed."""

    n = 10

    mat = np.zeros((n, n + 1))

    with np.testing.assert_raises_regex(Exception, 'square'):
        svd.svd(mat)

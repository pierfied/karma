import numpy as np
from lmapr.tools import cov


def test_cl2xi_zero():
    """Tests that the cl2xi_theta returns zero for zero input power spectrum."""

    cl = np.zeros(1000)
    theta = np.linspace(0, np.pi, 1000)

    xi = cov.cl2xi_theta(cl, theta)

    np.testing.assert_almost_equal(xi, 0)


def test_cl2xi_monopole():
    """Test that cl2xi has constant correlation over all separations for monopole power spectrum."""

    cl = np.zeros(1000)
    cl[0] = 1
    theta = np.linspace(0, np.pi, 1000)

    xi = cov.cl2xi_theta(cl, theta)

    np.testing.assert_almost_equal(xi, xi[0])


def test_cl2xi_dipole():
    """Test that cl2xi has opposite sign correlation at poles and zero at equator for dipole power spectrum."""

    cl = np.zeros(1000)
    cl[1] = 1
    theta = np.linspace(0, np.pi, 1000)

    xi = cov.cl2xi_theta(cl, theta)

    np.testing.assert_almost_equal(xi[0], -xi[-1])
    np.testing.assert_almost_equal(cov.cl2xi_theta(cl, np.pi / 2), 0)

def test_cl2cov_zero():
    """Test that covariance matrix is all zeros when power spectrum is zero."""

    nside = 4
    cl = np.zeros(1000)

    cov_mat = cov.cl2cov_mat(cl, nside)

    np.testing.assert_almost_equal(cov_mat, 0)


def test_cl2cov_monopole():
    """Test that covariance matrix is constant with monopole power spectrum."""

    nside = 4
    cl = np.zeros(1000)
    cl[0] = 1

    cov_mat = cov.cl2cov_mat(cl, nside)

    np.testing.assert_almost_equal(cov_mat, cov_mat[0,0])

def test_cl2cov_dipole():
    """Test that covariance matrix has opposite signs at poles with dipole power spectrum."""

    nside = 4
    cl = np.zeros(1000)
    cl[1] = 1

    cov_mat = cov.cl2cov_mat(cl, nside)

    # Using the second to last pixel for antipodal point due to geometry of Healpix.
    np.testing.assert_almost_equal(cov_mat[0,0], -cov_mat[0,-2])

def test_cl2cov_truncate():
    """Test that cl2cov will truncate l modes greater than lmax=0 and will return monopole results."""

    nside = 4
    cl = np.ones(1000)

    cov_mat = cov.cl2cov_mat(cl, nside, lmax=0)

    np.testing.assert_almost_equal(cov_mat, cov_mat[0,0])
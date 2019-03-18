import numpy as np
import healpy as hp
from lmapr.tools import transformations


def test_k2g_zero():
    """Test that transformation of a zero convergence map is a zero shear map."""

    nside = 16
    npix = hp.nside2npix(nside)

    k = np.zeros(npix)

    g1, g2 = transformations.conv2shear(k)

    np.testing.assert_almost_equal(g1, 0)
    np.testing.assert_almost_equal(g2, 0)


def test_g2k_zero():
    """Test that transformation of a zero shear map is a zero convergence map."""

    nside = 16
    npix = hp.nside2npix(nside)

    g1 = np.zeros(npix)
    g2 = np.zeros(npix)

    k = transformations.shear2conv(g1, g2)

    np.testing.assert_almost_equal(k, 0)


def test_k2g2k_rand():
    """Test that transformation of k->g->k recovers the input convergence map."""

    nside = 16
    npix = hp.nside2npix(nside)
    lmax = 32

    k = np.random.standard_normal(npix)
    k = hp.smoothing(k, lmax=lmax, verbose=False)
    k = hp.remove_monopole(k)
    k = hp.remove_dipole(k)

    g1, g2 = transformations.conv2shear(k, lmax)

    k_recov = transformations.shear2conv(g1, g2, lmax)

    np.testing.assert_almost_equal(k, k_recov, decimal=3)


def test_g2k_mismatch():
    """Test that shear2conv raises an error when g1 and g2 aren't the same length."""

    g1 = np.zeros(100)
    g2 = np.zeros(101)

    with np.testing.assert_raises(Exception):
        transformations.shear2conv(g1, g2)


def test_g2k_g1_dim():
    """Test that shear2conv raises an error when g1 is not a 1D array."""

    g1 = np.zeros((100, 100))
    g2 = np.zeros(100)

    with np.testing.assert_raises(Exception):
        transformations.shear2conv(g1, g2)


def test_g2k_g2_dim():
    """Test that shear2conv raises an error when g2 is not a 1D array."""

    g1 = np.zeros(100)
    g2 = np.zeros((100, 100))

    with np.testing.assert_raises(Exception):
        transformations.shear2conv(g1, g2)


def test_k2g_dim():
    """Test that conv2shear raises an error when k is not a 1D array."""

    k = np.zeros((100,100))

    with np.testing.assert_raises(Exception):
        transformations.conv2shear(k)

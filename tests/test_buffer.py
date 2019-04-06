import numpy as np
import healpy as hp
from karma.tools import mask


def test_add_buffer_empty_mask():
    """Test that add_buffer doesn't alter a map with an empty mask."""

    nside = 16
    npix = hp.nside2npix(nside)

    m = np.zeros(npix, dtype=bool)

    buffered_m = mask.add_buffer(m, 1)

    np.testing.assert_almost_equal(buffered_m, 0)


def test_add_buffer_full_mask():
    """Test that add_buffer doesn't alter a map with a full-sky mask."""

    nside = 16
    npix = hp.nside2npix(nside)

    m = np.ones(npix, dtype=bool)

    buffered_m = mask.add_buffer(m, 1)

    np.testing.assert_almost_equal(buffered_m, 1)


def test_add_buffer_one_pix():
    """Test that add_buffer does add a buffer of one pixel to a map with one pixel in the mask."""

    nside = 16
    npix = hp.nside2npix(nside)

    m = np.zeros(npix, dtype=bool)
    pix_ind = np.random.choice(npix, 1)
    m[pix_ind] = 1

    buffer = 1
    buffered_m = mask.add_buffer(m, buffer)

    np.testing.assert_array_less(m.sum(), buffered_m.sum())

    theta, phi = hp.pix2ang(nside, np.arange(npix)[buffered_m])
    ang_coord = np.stack([theta, phi], axis=0)

    ang_dist = hp.rotator.angdist(np.stack(hp.pix2ang(nside, pix_ind)).ravel(), ang_coord)

    avg_sep = hp.nside2resol(nside)
    np.testing.assert_array_less(ang_dist, 2 * buffer * avg_sep)


def test_add_buffer_ten_pix():
    """Test that add_buffer does add a buffer of ten pixels to a map with one pixel in the mask."""

    nside = 16
    npix = hp.nside2npix(nside)

    m = np.zeros(npix, dtype=bool)
    pix_ind = np.random.choice(npix, 1)
    m[pix_ind] = 1

    buffer = 10
    buffered_m = mask.add_buffer(m, buffer)

    np.testing.assert_array_less(m.sum(), buffered_m.sum())

    theta, phi = hp.pix2ang(nside, np.arange(npix)[buffered_m])
    ang_coord = np.stack([theta, phi], axis=0)

    ang_dist = hp.rotator.angdist(np.stack(hp.pix2ang(nside, pix_ind)).ravel(), ang_coord)

    avg_sep = hp.nside2resol(nside)
    np.testing.assert_array_less(ang_dist, 2 * buffer * avg_sep)


def test_add_buffer_dim():
    """Test that add_buffer raises an exception when mask is not a 1D array."""

    m = False

    with np.testing.assert_raises_regex(Exception, '1D'):
        mask.add_buffer(m, 1)

    m = np.zeros((10, 10), dtype=bool)

    with np.testing.assert_raises_regex(Exception, '1D'):
        mask.add_buffer(m, 1)


def test_add_buffer_int():
    """Test that add_buffer raises an exception when buffer is not an int."""

    m = np.zeros(10)

    with np.testing.assert_raises_regex(Exception, 'int'):
        mask.add_buffer(m, 0.1)

import numpy as np
import healpy as hp
from tqdm import tqdm


def cl2xi_theta(cl, theta):
    """Angular correlation function at separation theta from power spectrum.

    Computes the covariance between pixels at separation of theta from provided power spectrum.
    See https://arxiv.org/pdf/1602.08503.pdf equation 20.

    :param cl: Power spectrum.
    :type cl: array-like (float)
    :param theta: Separation angle in radians.
    :type theta: float
    :return: xi(theta) - Angular correlation at separation theta.
    :rtype: float
    """

    # Get array of l values.
    ells = np.arange(0, len(cl))

    # Compute xi(theta) using Legendre polynomials.
    xi = 1 / (4 * np.pi) * np.polynomial.legendre.legval(np.cos(theta), (2 * ells + 1) * cl)

    return xi


def cl2cov_mat(cl, nside, indices=None, lmax=None, ninterp=10000):
    """Covariance matrix from power spectrum.

    Computes the covariance matrix for the requested pixels from the provided power spectrum.

    :param cl: Power spectrum. Will truncate if len(cl) > lmax + 1.
    :type cl: array-like (float)
    :param nside: Healpix nside parameter.
    :type nside: int
    :param indices: Array of Healpix pixel numbers to compute covariance matrix for. Default: All pixels in the sky.
    :type indices: array-like (int)
    :param lmax: Maximum l mode to include in power spectrum. Default: len(cl) - 1
    :type lmax: float
    :param ninterp: Number of interpolation points for correlation function between 0 and pi. Default: 10,000
    :type ninterp: int
    :return: Covariance matrix.
    :rtype: array-like (float)
    """

    # Set lmax if not already set.
    if lmax is None:
        lmax = len(cl)

    # Truncate cl if necessary.
    if len(cl) > lmax + 1:
        input_cl = cl[:lmax + 1]
    else:
        input_cl = cl

    # If indices is not set default to all pixels.
    if indices is None:
        indices = np.arange(hp.nside2npix(nside))

    # Get the number of pixels.
    npix = len(indices)

    # Get angular coordinates for each pixel.
    theta, phi = hp.pix2ang(nside, indices)
    ang_coord = np.stack([theta, phi])

    # Calculate matrix of separations between pixels.
    ang_sep = np.zeros([npix, npix])
    for i in tqdm(range(npix), desc='Computing angular separations'):
        ang_sep[i, :] = hp.rotator.angdist(ang_coord[i], ang_coord)

    # Construct interpolation points for the angular correlation function.
    theta_interp = np.linspace(0, np.pi, ninterp)
    xi_interp = cl2xi_theta(input_cl, theta_interp)

    # Compute covariance matrix using linear interpolation.
    cov = np.interp(ang_sep, theta_interp, xi_interp)

    return cov

import ctypes
import os
import numpy as np
from . import tools
import healpy as hp


class HMCArgs(ctypes.Structure):
    _fields_ = [
        ('log_likelihood', ctypes.c_void_p),
        ('likelihood_args', ctypes.c_void_p),
        ('num_params', ctypes.c_int),
        ('num_burn', ctypes.c_int),
        ('num_burn_steps', ctypes.c_int),
        ('burn_epsilon', ctypes.c_double),
        ('num_samples', ctypes.c_int),
        ('num_samp_steps', ctypes.c_int),
        ('samp_epsilon', ctypes.c_double),
        ('x0', ctypes.POINTER(ctypes.c_double)),
        ('m', ctypes.POINTER(ctypes.c_double)),
        ('sigma_p', ctypes.POINTER(ctypes.c_double)),
        ('verbose', ctypes.c_int)
    ]


class KarmaArgs(ctypes.Structure):
    _fields_ = [
        ('mask_npix', ctypes.c_long),
        ('buffer_npix', ctypes.c_long),
        ('num_vecs', ctypes.c_long),
        ('mu', ctypes.c_double),
        ('shift', ctypes.c_double),
        ('s', ctypes.POINTER(ctypes.c_double)),
        ('u', ctypes.POINTER(ctypes.c_double)),
        ('k2g1', ctypes.POINTER(ctypes.c_double)),
        ('k2g2', ctypes.POINTER(ctypes.c_double)),
        ('g1_obs', ctypes.POINTER(ctypes.c_double)),
        ('g2_obs', ctypes.POINTER(ctypes.c_double)),
        ('sigma_g1', ctypes.POINTER(ctypes.c_double)),
        ('sigma_g2', ctypes.POINTER(ctypes.c_double)),
        ('sigma_gh1', ctypes.c_double),
        ('sigma_gh2', ctypes.c_double),
    ]


class SampleChain(ctypes.Structure):
    _fields_ = [
        ('num_samples', ctypes.c_int),
        ('num_params', ctypes.c_int),
        ('accept_rate', ctypes.c_double),
        ('samples', ctypes.POINTER(ctypes.POINTER(ctypes.c_double))),
        ('log_likelihoods', ctypes.POINTER(ctypes.c_double))
    ]


class KarmaSampler:
    mu = None
    shift = None
    s = None
    u = None
    k2g1 = None
    k2g2 = None
    g1_obs = None
    g2_obs = None
    sigma_g = None

    def __init__(self, g1_obs, g2_obs, sigma_g1, sigma_g2, sigma_gh1, sigma_gh2):
        """Initializer for karma sampler class.

        :param g1_obs: First observed shear component.
        :type g1_obs: array-like (float)
        :param g2_obs: Second observed shear component.
        :type g2_obs: array-like (float)
        :param sigma_g: Per pixel error in shear observations.
        :type sigma_g: array-like (float)
        """

        self.g1_obs = g1_obs
        self.g2_obs = g2_obs
        self.sigma_g1 = sigma_g1
        self.sigma_g2 = sigma_g2
        self.sigma_gh1 = sigma_gh1
        self.sigma_gh2 = sigma_gh2

    def set_lognorm_params(self, mu, shift, cov=None, rcond=None, u=None, s=None):
        """Setter for the parameters of the lognormal prior.

        :param mu: Mu parameter of the lognormal distribution.
        :type mu: float
        :param shift: Shift parameter of the lognormal distribution.
        :type shift: float
        :param cov: Covariance matrix. If set will compute SVD of the matrix. Default: None
        :type cov: array-like (float)
        :param rcond: SVD truncation parameter used only if cov is set. If rcond is a float will truncate SVD to keep
                      only vectors with s/s_max > rcond. If rcond is an int will truncate SVD to keep the first rcond
                      vectors. Default: None
        :type rcond: float or int
        :param u: Pre-computed rotation matrix. Only be used if cov is not set. Default: None
        :type u: array-like (float)
        :param s: Pre-computed singular values. Only be used if cov is not set. Default: None
        :type s: array-like (float)
        """

        # Set the mu and shift parameters of the lognormal distribution.
        self.mu = mu
        self.shift = shift

        # Check if the SVD of the covariance needs to be computed.
        if cov is None:
            # Set SVD results if precomputed.
            self.u = u
            self.s = s
        else:
            # Perform SVD on the covariance if set.
            u, s, _ = tools.svd.svd(cov)

            # Perform truncation as specified if requested.
            if type(rcond) is float:
                good_vecs = s / s[0] > rcond

                s = s[good_vecs]
                u = u[:, good_vecs]
            elif type(rcond) is int:
                s = s[:rcond]
                u = u[:, :rcond]

            self.s = s
            self.u = u

    def set_k2g_mats(self, nside=None, mask=None, buffered_mask=None, lmax=None, k2g1=None, k2g2=None):
        """Setter for the kappa -> gamma transformation matrices.

        :param nside: Healpix nside parameter. Only used if k2g1 is not set. Default: None
        :type nside: int
        :param mask: Masked region of the sky to compute the matrices for. Default: All pixels in the sky.
        :type mask: array-like (bool)
        :param buffered_mask: Masked region with buffer. If set and mask is set k2g will have
                              shape mask x buffered_mask. Default: None
        :type buffered_mask: array-like (bool)
        :param lmax: Maximum l mode to include in transformation. Default: 3*nside - 1
        :type lmax: int
        :param k2g1: Pre-computed kappa -> gamma matrix for first shear component. If not set will compute k2g matrices.
                     Default: None
        :type k2g1: array-like (float)
        :param k2g2: Pre-computed kappa -> gamma matrix for second shear component.
                     Default: None
        :type k2g2: array-like (float)
        """

        # Check if k2g matrices need to be computed.
        if k2g1 is not None:
            self.k2g1 = k2g1
            self.k2g2 = k2g2
        else:
            # Get the indices of interest for the k2g matrices.
            indices = np.arange(hp.nside2npix(nside))

            # If mask or buffered_mask is set, compute for only those indices.
            if mask is not None and buffered_mask is not None:
                indices = indices[buffered_mask]
            elif mask is not None:
                indices = indices[mask]

            # Compute the matrices.
            k2g1, k2g2 = tools.transformations.conv2shear_mats(nside, indices, lmax)

            # If buffered_mask is set, truncate the "output" of the transformation to only the mask region.
            if mask is not None and buffered_mask is not None:
                k2g1 = k2g1[mask, :]
                k2g2 = k2g2[mask, :]

            self.k2g1 = k2g1
            self.k2g2 = k2g2

    def sample(self, num_burn, num_burn_steps, burn_epsilon, num_samples, num_samp_steps, samp_epsilon, k0=None,
               verbose=True):
        """Run the HMC sampler on the likelihood.

        :param num_burn: Number of samples for burn-in.
        :type num_burn: int
        :param num_burn_steps: Number of leapfrog steps per burn-in sample.
        :type num_burn_steps: int
        :param burn_epsilon: Step-size for leapfrog during burn-in.
        :type burn_epsilon: float
        :param num_samples: Number of samples.
        :type num_samples: int
        :param num_samp_steps: Number of leapfrog steps per sample.
        :type num_samp_steps: int
        :param samp_epsilon: Step-size for leapfrog during sampling.
        :type samp_epsilon: float
        :param k0: Initial kappa parameter values (optional). Default: Randomly initialized.
        :type k0: array-like (float)
        :param verbose: Have HMC print out at each step. Default: True
        :type verbose: bool
        :returns Acceptance rate, chains in kappa, and log-likelihoods.
        :rtype Tuple (float, array-like (float), array-like (float))
        """

        # Check that all quantities are set.
        if (self.mu is None or self.shift is None or self.s is None or self.u is None or self.k2g1 is None
                or self.k2g2 is None or self.g1_obs is None or self.g2_obs is None or self.sigma_g is None):
            raise Exception('One or more of the following parameters is not set: '
                            'mu, shift, s, u, k2g1, k2g2, g1_obs, g2_obs, sigma_g')

        # Convert array-like quantities to ndarrays.
        s = np.asarray(self.s)
        u = np.asarray(self.u)
        k2g1 = np.asarray(self.k2g1)
        k2g2 = np.asarray(self.k2g2)
        g1_obs = np.asarray(self.g1_obs)
        g2_obs = np.asarray(self.g2_obs)
        sigma_g1 = np.asarray(self.sigma_g1)
        sigma_g2 = np.asarray(self.sigma_g2)

        # Check dimensions of inputs.
        if s.ndim != 1:
            raise Exception('Singular values (s) must be a 1D array.')
        if u.ndim != 2:
            raise Exception('Rotation matrix (u) must be a 2D array.')
        if k2g1.ndim != 2 or k2g2.ndim != 2:
            raise Exception('Kappa -> gamma transformation matrices (k2g1, k2g2) must be 2D arrays.')
        if g1_obs.ndim != 1 or g2_obs.ndim != 1:
            raise Exception('Observed gamma values (g1_obs, g2_obs) must be 1D arrays.')
        if sigma_g1.ndim != 1 or sigma_g2.ndim != 1:
            raise Exception('Sigma gamma values (sigma_g) must be a 1D array.')

        # Check for inconsistencies in the shapes of the inputs.
        num_vecs = len(s)
        buffer_npix = u.shape[0]
        mask_npix = len(g1_obs)
        if (k2g1.shape != k2g2.shape or g1_obs.shape != g2_obs.shape or g1_obs.shape != sigma_g1.shape
                or u.shape[1] != num_vecs or k2g1.shape[0] != mask_npix or k2g1.shape[1] != buffer_npix
                or sigma_g1.shape != sigma_g2.shape):
            raise Exception('Input shapes are inconsistent.')

        # If k0 is set compute x0 values.
        if k0 is not None:
            # Convert to ndarray.
            k0 = np.asarray(k0)

            # Check the dimensions and shape of k0.
            if k0.ndim != 1:
                raise Exception('Initial kappa values must be a 1D array.')
            if len(k0) != buffer_npix:
                raise Exception('Length of initial kappa values is inconsistent.')

            # Compute y values.
            y0 = np.log(self.shift + k0)

            # Transform y values into diagonal basis.
            x0 = u.T @ (y0 - self.mu)
        else:
            # Otherwise randomly initialize x0.
            # x0 = np.random.standard_normal(num_vecs) * np.sqrt(s)
            x0 = np.random.standard_normal(num_vecs + 2 * mask_npix) * np.sqrt(
                np.concatenate([s, self.sigma_gh1 ** 2, self.sigma_gh2 ** 2]))

        # Calculate the optimal scaling of the mass and momenta for HMC.
        m = 1 / np.concatenate([s, self.sigma_gh1 ** 2, self.sigma_gh2 ** 2])
        sigma_p = np.sqrt(m)

        # Ensure that all quantities are contiguous arrays in memory.
        s = np.ascontiguousarray(s)
        u = np.ascontiguousarray(u)
        k2g1 = np.ascontiguousarray(k2g1)
        k2g2 = np.ascontiguousarray(k2g2)
        g1_obs = np.ascontiguousarray(g1_obs)
        g2_obs = np.ascontiguousarray(g2_obs)
        sigma_g1 = np.ascontiguousarray(sigma_g1)
        sigma_g2 = np.ascontiguousarray(sigma_g2)
        x0 = np.ascontiguousarray(x0)
        m = np.ascontiguousarray(m)
        sigma_p = np.ascontiguousarray(sigma_p)

        # Create the args struct for the sampler.
        d_ptr = ctypes.POINTER(ctypes.c_double)
        hmc_args = HMCArgs()
        hmc_args.num_params = num_vecs
        hmc_args.num_burn = num_burn
        hmc_args.num_burn_steps = num_burn_steps
        hmc_args.burn_epsilon = burn_epsilon
        hmc_args.num_samples = num_samples
        hmc_args.num_samp_steps = num_samp_steps
        hmc_args.samp_epsilon = samp_epsilon
        hmc_args.x0 = x0.ctypes.data_as(d_ptr)
        hmc_args.m = m.ctypes.data_as(d_ptr)
        hmc_args.sigma_p = sigma_p.ctypes.data_as(d_ptr)
        hmc_args.verbose = int(verbose)

        # Create the args struct for the likelihood function.
        karma_args = KarmaArgs()
        karma_args.mask_npix = mask_npix
        karma_args.buffer_npix = buffer_npix
        karma_args.num_vecs = num_vecs
        karma_args.mu = self.mu
        karma_args.shift = self.shift
        karma_args.s = s.ctypes.data_as(d_ptr)
        karma_args.u = u.ctypes.data_as(d_ptr)
        karma_args.k2g1 = k2g1.ctypes.data_as(d_ptr)
        karma_args.k2g2 = k2g2.ctypes.data_as(d_ptr)
        karma_args.g1_obs = g1_obs.ctypes.data_as(d_ptr)
        karma_args.g2_obs = g2_obs.ctypes.data_as(d_ptr)
        karma_args.sigma_g1 = sigma_g1.ctypes.data_as(d_ptr)
        karma_args.sigma_g2 = sigma_g2.ctypes.data_as(d_ptr)
        karma_args.sigma_gh1 = self.sigma_gh1
        karma_args.sigma_gh2 = self.sigma_gh2

        # Load the KARMA library.
        lib_path = os.path.join(os.path.dirname(__file__), 'libkarma.so')
        karma_lib = ctypes.cdll.LoadLibrary(lib_path)

        # Define the call to the sampler driver.
        karma_sample = karma_lib.karma_sample
        karma_sample.argtypes = [HMCArgs, KarmaArgs]
        karma_sample.restype = SampleChain

        # Run the sampler.
        results = karma_sample(hmc_args, karma_args)

        # Get the results from the returned struct.
        accept_rate = results.accept_rate
        chain = np.stack(
            [np.ctypeslib.as_array(results.samples[i], shape=(num_vecs + 2 * mask_npix,)) for i in range(num_samples)])
        logp = np.ctypeslib.as_array(results.log_likelihoods, shape=(num_samples,))

        # Trim the high-frequency parameters from the chain.
        chain = chain[:, :num_vecs]

        # Convert chain from diagonal basis to kappa samples.
        chain = chain @ u.T + self.mu
        chain = np.exp(chain) - self.shift

        return accept_rate, chain, logp

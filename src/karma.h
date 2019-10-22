//
// Created by pierfied on 4/4/19.
//

#ifndef KARMA_KARMA_H
#define KARMA_KARMA_H

#include <hmc.h>

/**
 * @brief Struct containing the args for the likelihood function.
 *
 * @param mask_npix Number of pixels in the mask region.
 * @param buffer_npix Number of pixels in the full mask + buffer region.
 * @param num_vecs Number of singular vectors included in diagonalization.
 * @param mu Mean parameter of the lognormal model.
 * @param shift Shift parameter of the lognormal model.
 * @param s Singular values of covariance matrix with length num_vecs.
 * @param u Rotation matrix to diagonalize covariance matrix with shape buffer_npix x num_vecs.
 * @param k2g1 Matrix to perform kappa -> gamma_1 component transformation with shape mask_npix x buffer_npix.
 * @param k2g2 Matrix to perform kappa -> gamma_2 component transformation with shape mask_npix x buffer_npix.
 * @param g1_obs Observed first shear components with length mask_npix.
 * @param g2_obs Observed second shear components with length mask_npix.
 * @param sigma_g Error in shear observations with length mask_npix.
 */
typedef struct KarmaArgs_struct {
    long mask_npix;
    long buffer_npix;
    long num_vecs;
    double mu;
    double shift;
    double *s;
    double *u;
    double *k2g1;
    double *k2g2;
    double *g1_obs;
    double *g2_obs;
    double *sigma_g;
    double *sgr;
    double *ugr;
} KarmaArgs;

/**
 * @brief Runs the HMC sampler on the full posterior.
 *
 * @param hmc_args HMCArgs struct containing arguments for the HMC sampler.
 *                 Likelihood function and args will be set here and do not need to be set beforehand.
 * @param karma_args KarmaArgs struct with arguments for the likelihood function.
 * @return SampleChain struct with the Markov chain of the sample run.
 */
SampleChain karma_sample(HMCArgs hmc_args, KarmaArgs karma_args);

/**
 * @brief Calculates the map likelihood and gradient to be used by the HMC sampler.
 *
 * @param x Parameter values in the diagonal basis.
 * @param args_ptr Pointer to a KarmaArgs struct.
 * @return Hamiltonian struct containing the log-likelihood and gradient.
 */
Hamiltonian karma_likelihood(double *x, void *args_ptr);

#endif //KARMA_KARMA_H

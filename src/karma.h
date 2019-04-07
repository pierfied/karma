//
// Created by pierfied on 4/4/19.
//

#ifndef KARMA_KARMA_H
#define KARMA_KARMA_H

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
} KarmaArgs;

#endif //KARMA_KARMA_H

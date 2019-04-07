//
// Created by pierfied on 4/4/19.
//

#include "karma.h"
#include <hmc.h>
#include <stdlib.h>
#include <math.h>

/**
 * @brief Calculates the map likelihood and gradient to be used by the HMC sampler.
 *
 * @param x Parameter values in the diagonal basis.
 * @param args_ptr Pointer to a KarmaArgs struct.
 * @return Hamiltonian struct containing the log-likelihood and gradient.
 */
Hamiltonian karma_likelihood(double *x, void *args_ptr) {
    // Cast the arg pointer to a KarmArgs pointer.
    KarmaArgs *args = (KarmaArgs *) args_ptr;

    // Get all parameters from the args struct for performance.
    long mask_npix = args->mask_npix;
    long buffer_npix = args->buffer_npix;
    long num_vecs = args->num_vecs;
    double mu = args->mu;
    double shift = args->shift;
    double *s = args->s;
    double *u = args->u;
    double *k2g1 = args->k2g1;
    double *k2g2 = args->k2g2;
    double *g1_obs = args->g1_obs;
    double *g2_obs = args->g2_obs;
    double *sigma_g = args->sigma_g;

    // Calculate the kappa and log parameter values from the diagonal basis parameters.
    double y[buffer_npix];
    double exp_y[buffer_npix];
    double kappa[buffer_npix];
#pragma omp parallel for
    for (long i = 0; i < buffer_npix; ++i) {
        y[i] = mu;

        // Compute y from the diagonal basis parameters.
        for (long j = 0; j < num_vecs; ++j) {
            y[i] += u[i * num_vecs + j] * x[j];
        }

        // Pre-calculate exp(y).
        exp_y[i] = exp(y[i]);

        // Calculate kappa.
        kappa[i] = exp_y[i] - shift;
    }

    // Compute the shear components from the kappa values via a linear transformation.
    double g1[mask_npix];
    double g2[mask_npix];
#pragma omp parallel for
    for (long i = 0; i < mask_npix; ++i) {
        g1[i] = 0;
        g2[i] = 0;
        
        for (long j = 0; j < buffer_npix; ++j) {
            long ind = i * buffer_npix + j;

            g1[i] += k2g1[ind] * kappa[j];
            g2[i] += k2g2[ind] * kappa[j];
        }
    }

    // Compute the contribution of each pixel to the shear likelihood and the gradients df/dg.
    double f1_g1[mask_npix];
    double f2_g2[mask_npix];
    double df1_dg1[mask_npix];
    double df2_dg2[mask_npix];
#pragma omp parallel for
    for (long i = 0; i < mask_npix; ++i) {
        double delta_g1_i = g1[i] - g1_obs[i];
        double delta_g2_i = g2[i] - g2_obs[i];

        // Calculate the gradient df/dg.
        double var_g_i = sigma_g[i] * sigma_g[i];
        df1_dg1[i] = delta_g1_i / var_g_i;
        df2_dg2[i] = delta_g2_i / var_g_i;

        // Calculate the shear likelihood contribution.
        f1_g1[i] = df1_dg1[i] * delta_g1_i;
        f2_g2[i] = df2_dg2[i] * delta_g2_i;
    }

    // Calculate the gradients df/dy via chain rule.
    double df1_dy[buffer_npix];
    double df2_dy[buffer_npix];
#pragma omp parallel for
    for (long i = 0; i < buffer_npix; ++i) {
        df1_dy[i] = 0;
        df2_dy[i] = 0;

        for (long j = 0; j < mask_npix; ++j) {
            long ind = j * buffer_npix + i;

            df1_dy[i] += df1_dg1[j] * k2g1[ind];
            df2_dy[i] += df2_dg2[j] * k2g2[ind];
        }

        df1_dy[i] *= exp_y[i];
        df2_dy[i] *= exp_y[i];
    }

    // Calculate the gradients df/dx via chain rule and the total gradient dlnP/dx.
    double *grad = malloc(sizeof(double) * num_vecs);
#pragma omp parallel for
    for (long i = 0; i < num_vecs; ++i) {
        double df1_dx_i = 0;
        double df2_dx_i = 0;

        // Calculate the gradients df/dx.
        for (long j = 0; j < buffer_npix; ++j) {
            long ind = j * num_vecs + i;

            df1_dx_i += df1_dy[j] * u[ind];
            df2_dx_i += df2_dy[j] * u[ind];
        }

        // Calculate the total gradient dlnP/dx.
        grad[i] = -(x[i] / s[i] + df1_dx_i + df2_dx_i);
    }

    // Compute the total log-likelihood.
    double log_p = 0;
    for (long i = 0; i < num_vecs; ++i) {
        log_p += (x[i] * x[i]) / s[i] + f1_g1[i] + f2_g2[i];
    }
    log_p *= -0.5;

    // Create the Hamiltonian struct.
    Hamiltonian likelihood;
    likelihood.log_likelihood = log_p;
    likelihood.grad = grad;

    return likelihood;
}
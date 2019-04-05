//
// Created by pierfied on 4/4/19.
//

#ifndef KARMA_SVD_H
#define KARMA_SVD_H

/**
 * @brief Struct containing SVD results of a square matrix A = U * S * VT.
 *
 * @var len Length of the matrix in the unrotated space.
 * @var num_vecs Number of singular vectors/values.
 * @var U Pointer to left rotation matrix U with shape len x num_vecs.
 * @var S Pointer to the singular values S with shape num_vecs.
 * @var VT Pointer to the right rotation matrix VT with shape num_vecs x len.
 * @var info Status returned by LAPACK.
 */
typedef struct USVT_struct {
    long len;
    long num_vecs;
    double *U;
    double *S;
    double *VT;
    long info;
} USVT;

/**
 * @brief Computes full SVD of a dense square matrix.
 *
 * @param len The length of one side of the matrix.
 * @param mat Pointer to the matrix.
 * @return USVT struct containing svd results.
 */
USVT svd(long len, double *mat);

#endif //KARMA_SVD_H

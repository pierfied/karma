//
// Created by pierfied on 4/4/19.
//

#include "svd.h"
#include <lapacke.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * @brief Reads a matrix from stdin, runs SVD, and returns the result through stdout.
 *
 * Assumes input from stdin is formatted as:
 * First 8 bytes will be length of the matrix along one of its dimensions.
 * Remaining (8 * length * length) bytes will be the double-precision matrix.
 *
 * Output via stdout is formatted as:
 * First (8 * length * length) bytes will be the left rotation matrix U as double-precision.
 * Next (8 * length) bytes will be the singular values S as double-precision.
 * Final (8 * length * length) bytes will be the right rotation matrix VT as double-precision.
 *
 * @return LAPACK exit status.
 */
int main() {
    // Change the stdin and stdout to binary mode.
    FILE *f;
    f = freopen(NULL, "rb", stdin);
    f = freopen(NULL, "wb", stdout);

    // Read how long the matrix is going to be.
    long len;
    int res;
    res = fread(&len, sizeof(long), 1, stdin);

    // Read the matrix.
    double *mat = malloc(sizeof(double) * len * len);
    res = fread(mat, sizeof(double), len * len, stdin);

    // Perform SVD.
    USVT result = svd(len, mat);

    // Write the results to stdout.
    fwrite(result.U, sizeof(double), len * len, stdout);
    fwrite(result.S, sizeof(double), len, stdout);
    fwrite(result.VT, sizeof(double), len * len, stdout);

    // Free the matrices.
    free(mat);
    free(result.U);
    free(result.S);
    free(result.VT);

    return result.info;
}

USVT svd(long len, double *mat) {
    // Allocate the arrays for the singular vectors/values.
    double *U = malloc(sizeof(double) * len * len);
    double *S = malloc(sizeof(double) * len);
    double *VT = malloc(sizeof(double) * len * len);

    // Run the LAPACK SVD routine.
    long info = LAPACKE_dgesdd(LAPACK_ROW_MAJOR, 'A', len, len, mat, len, S, U, len, VT, len);

    // Store the results in the return struct.
    USVT result;
    result.len = len;
    result.num_vecs = len;
    result.U = U;
    result.S = S;
    result.VT = VT;
    result.info = info;

    return result;
}
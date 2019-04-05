//
// Created by pierfied on 4/4/19.
//

#include "svd.h"
#include <lapacke.h>

USVT svd(long len, double *mat){
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
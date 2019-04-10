import numpy as np
import subprocess
import os


def svd(mat):
    """Perform SVD on a square matrix A = U * S * VT.

    Uses OpenBLAS LAPACKE dgesdd routine to perform SVD.

    :param mat: A square matrix.
    :type mat: array-like (float)
    :return: U, S, and VT (left rotation matrix, singular values, right rotation matrix respectively).
    :rtype: Tuple of array-like (float)
    """

    # Convert array-like input to ndarray.
    mat = np.asarray(mat)

    # Check input size.
    if mat.ndim != 2:
        raise Exception('Input must be a 2D matrix.')
    if mat.shape[0] != mat.shape[1]:
        raise Exception('Input matrix must be square.')

    mat_len = len(mat)

    # Load the KARMA SVD executable.
    exe_path = os.path.join(os.path.dirname(__file__), '../karma_svd')
    proc = subprocess.Popen(exe_path, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    # Pass the matrix to the SVD routine via stdin.
    proc.stdin.write(mat_len.to_bytes(8, 'little'))
    proc.stdin.write(mat.tobytes())
    proc.stdin.close()

    # Get the results from the SVD routine via stdout.
    data = np.frombuffer(proc.stdout.read())

    # Extract U.
    start = 0
    end = mat_len * mat_len
    U = data[start:end].reshape([mat_len, mat_len])

    # Extract S.
    start = end
    end += mat_len
    S = data[start:end]

    # Extract VT.
    start = end
    end += mat_len * mat_len
    VT = data[start:end].reshape([mat_len, mat_len])

    # Get the LAPACK return status.
    info = proc.wait()

    if info != 0:
        raise Exception('LAPACK returned status %d.' % info)

    return U, S, VT

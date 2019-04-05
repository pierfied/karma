import numpy as np
import ctypes
import os


class USVT(ctypes.Structure):
    """USVT struct class for ctypes."""

    _fields_ = [
        ('len', ctypes.c_long),
        ('num_vecs', ctypes.c_long),
        ('U', ctypes.POINTER(ctypes.c_double)),
        ('S', ctypes.POINTER(ctypes.c_double)),
        ('VT', ctypes.POINTER(ctypes.c_double)),
        ('info', ctypes.c_long)
    ]

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

    # Load the KARMA library.
    lib_path = os.path.join(os.path.dirname(__file__), '../libkarma.so')
    print(lib_path)
    karma_lib = ctypes.cdll.LoadLibrary(lib_path)

    # Define the call to the c svd routine.
    karma_svd = karma_lib.svd
    karma_svd.argtypes = [ctypes.c_long, ctypes.POINTER(ctypes.c_double)]
    karma_svd.restype = USVT

    # Create the args to pass to the c svd routine.
    len = mat.shape[0]
    mat_copy = np.ascontiguousarray(mat.copy(), dtype=np.double)
    mat_copy_p = mat_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    # Run the c svd routine.
    usvt = karma_svd(len, mat_copy_p)

    if usvt.info != 0:
        raise Exception('LAPACK returned status %d.' % usvt.info)

    U = np.ctypeslib.as_array(usvt.U, shape=(len, len))
    S = np.ctypeslib.as_array(usvt.S, shape=(len,))
    VT = np.ctypeslib.as_array(usvt.VT, shape=(len, len))

    return U, S, VT

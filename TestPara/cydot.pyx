#cython: boundscheck=False, wraparound=False
import  numpy as np
cimport numpy as np

from cython.parallel cimport prange

def dot(np.ndarray[np.float32_t, ndim=2] a not None,
        np.ndarray[np.float32_t, ndim=2] b not None,
        np.ndarray[np.float32_t, ndim=2] out=None):
    """Naive O(N**3) 2D np.dot() implementation."""
    if out is None:
        out = np.empty((a.shape[0], b.shape[1]), dtype=a.dtype)
    if (a.shape[1] != b.shape[0] or
        out.shape[0] != a.shape[0] or out.shape[1] != b.shape[1]):
        raise ValueError("wrong shape")

    cdef Py_ssize_t i, j, k
    with nogil:
        for i in prange(a.shape[0]):
            for j in range(b.shape[1]):
                out[i,j] = 0
                for k in range(a.shape[1]):
                    out[i,j] += a[i,k] * b[k,j]
    return out
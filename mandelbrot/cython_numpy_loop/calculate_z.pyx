#cython: boundscheck=False
"""
Mandelbrot Cython
-----------------
This code uses complex numbers requiring a recent version of Cythnon > 0.11.2
Attribution: Didrik Pinte (Enthought) for first version
"""


from numpy import empty, zeros
cimport numpy as np

def calculate_z(np.ndarray[double, ndim=1] xs, np.ndarray[double, ndim=1] ys, int maxiter):
    """ Generate a mandelbrot set """
    cdef unsigned int i,j
    cdef unsigned int N = len(xs)
    cdef unsigned int M = len(ys)
    cdef double complex q
    cdef double complex z
    cdef int iteration
    
    cdef np.ndarray[int, ndim=2] d = empty(dtype='i', shape=(M, N))
    for j in range(M):
        for i in range(N):
            # create q without intermediate object (faster)
            q = xs[i] + ys[j]*1j
            z = 0+0j
            for iteration in range(maxiter):
                z = z*z + q
                if z.real*z.real + z.imag*z.imag > 4.0:  
                    break
            else:
                iteration = 0
            d[j,i] = iteration
    return d

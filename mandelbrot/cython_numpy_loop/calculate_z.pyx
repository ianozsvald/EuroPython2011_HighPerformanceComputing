#cython: boundscheck=False
"""
Mandelbrot Cython
-----------------
This code uses complex numbers requiring a recent version of Cythnon > 0.11.2
Attribution: Didrik Pinte (Enthought) for first version
"""


from numpy import empty, zeros
cimport numpy as np

cdef int mandelbrot_escape(double complex q, int maxiter, double complex z):
    """ Mandelbrot set escape time algorithm in real and complex components
    """
    cdef int i
    for i in range(maxiter):
        z = z*z + q
        if z.real*z.real + z.imag*z.imag > 4.0:  
            break
    else:
        i = 0
    return i

def calculate_z(np.ndarray[double, ndim=1] xs, np.ndarray[double, ndim=1] ys, int maxiter):
    """ Generate a mandelbrot set """
    cdef unsigned int i,j
    cdef unsigned int N = len(xs)
    cdef unsigned int M = len(ys)
    cdef double complex q
    cdef double complex z
    z = 0+0j
    
    cdef np.ndarray[int, ndim=2] d = empty(dtype='i', shape=(M, N))
    for j in range(M):
        for i in range(N):
            # create q without intermediate object (faster)
            q = xs[i] + ys[j]*1j
            # create a temporary complex (consistently a bit slower)
            #q = complex(xs[i], ys[j])
            d[j,i] = mandelbrot_escape(q, maxiter, z)
    return d

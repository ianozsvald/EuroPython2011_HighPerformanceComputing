
import numpy as np

def mandelbrot_escape(q, n, z):
    """ Mandelbrot set escape time algorithm in real and complex components
    """
    for i in range(n):
        z = z*z + q
        if z.real*z.real + z.imag*z.imag > 4.0:
           break
    else:
        i = 0 
    return i

def calculate_z(xs, ys, maxiter):
    """ Generate a mandelbrot set """
    N = len(xs)
    M = len(ys)
    z = 0+0j
    
    d = np.zeros((M, N)).astype(np.int)
    for j in range(M):
        for i in range(N):
            q = xs[i] + ys[j]*1j
            d[j,i] = mandelbrot_escape(q, maxiter, z)
    return d
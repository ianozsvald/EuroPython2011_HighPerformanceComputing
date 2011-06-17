
import numpy as np

import datetime # DEBUG

def mandelbrot_escape(q, n, z):
    """ Mandelbrot set escape time algorithm in real and complex components
    """
    for i in range(n):
        # straight math is faster!
        z = z*z + q
        # breaking the math out makes it run at half the speed!
        #if z.real*z.real + z.imag*z.imag > 4.0:
        if abs(z) > 2:
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
        print "CALC J", j, M, datetime.datetime.now()
        for i in range(N):
            q = xs[i] + ys[j]*1j
            d[j,i] = mandelbrot_escape(q, maxiter, z)
    return d

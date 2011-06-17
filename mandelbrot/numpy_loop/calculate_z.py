
import numpy as np

def mandelbrot_escape(c, n):
    """ Mandelbrot set escape time algorithm in real and complex components
    """
    z = 0 # DP : Ian does initialise it to zero, we do initialise it to c
    for i in range(n):
        z = z*z + c
        if z.real*z.real + z.imag*z.imag > 4.0:  # DP :  was >=
        #if abs(z) > 2.0:  # DP :  was >=
           break
    else:
        i = 0 # DP : was returning -1 when not enough iterations ...
    return i

def generate_mandelbrot(xs, ys, n):
    """ Generate a mandelbrot set """
    N = len(xs)
    M = len(ys)
    
    #np.ndarray[int, ndim=2] d = empty(dtype='i', shape=(M, N))
    d = np.zeros((M, N)).astype(np.int)
    for j in range(M):
        for i in range(N):
            z = xs[i] + ys[j]*1j
            d[j,i] = mandelbrot_escape(z, n)
    return d

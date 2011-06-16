# Use to show the speed difference between CPU and GPU calculations
# ian@ianozsvald.com July 2011

import sys
import datetime
import numpy as nm


import pycuda.driver as drv
import pycuda.autoinit
import numpy
from pycuda.compiler import SourceModule
from pycuda.elementwise import ElementwiseKernel
import pycuda.gpuarray as gpuarray

# area of space to investigate
x1, x2, y1, y2 = -2.13, 0.77, -1.3, 1.3

# calculate_z using a CUDA card, this defaults to float32 to support
# older CUDA devices, just edit two lines below lines to use float64s on 
# newer CUDA devices

# create an ElementwiseKernel using a block of C code, 'i' represents the item in the current array (where
# a rows of is are executed simultaneously). Only works when everything is updating a the same
# array in lockstep
complex_gpu = ElementwiseKernel(
        """pycuda::complex<float> *z, pycuda::complex<float> *q, int *iteration, int maxiter""",
            """for (int n=0; n < maxiter; n++) {z[i] = (z[i]*z[i])+q[i]; if (abs(z[i]) > 2.00f) {iteration[i]=n; z[i] = pycuda::complex<float>(); q[i] = pycuda::complex<float>();};};""",
        "complex5",
        preamble="""#include <pycuda-complex.hpp>""",
        keep=True)


def calculate_z_gpu_elementwise(q, maxiter, z):
    # convert complex128s (2*float64) to complex64 (2*float32) so they run
    # on older CUDA cards like the one in my MacBook. To use float64 doubles
    # just edit these two lines
    complex_type = nm.complex64 # or nm.complex128 on newer CUDA devices
    #float_type = nm.float32 # or nm.float64 on newer CUDA devices
    output = nm.resize(nm.array(0,), q.shape)
    q_gpu = gpuarray.to_gpu(q.astype(complex_type))
    z_gpu = gpuarray.to_gpu(z.astype(complex_type))
    iterations_gpu = gpuarray.to_gpu(output) 
    print "maxiter gpu", maxiter
    # the for loop and complex calculations are all done on the GPU
    # we bring the iterations_gpu array back to determine pixel colours later
    complex_gpu(z_gpu, q_gpu, iterations_gpu, maxiter)

    iterations = iterations_gpu.get()
    return iterations


def calculate(show_output):
    # make a list of x and y values
    # xx is e.g. -2.13,...,0.712
    xx = nm.arange(x1, x2, (x2-x1)/w*2)
    # yy is e.g. 1.29,...,-1.24
    yy = nm.arange(y2, y1, (y1-y2)/h*2) * 1j
    # we see a rounding error for arange on yy with h==1000
    # so here I correct for it
    if len(yy) > h / 2.0:
        yy = yy[:-1]
    assert len(xx) == w / 2.0
    assert len(yy) == h / 2.0

    print "xx and yy have length", len(xx), len(yy)

    # yy will become 0+yyj when cast to complex128 (2 * 8byte float64) same as Python float 
    yy = yy.astype(nm.complex128)
    # create q as a square matrix initially of complex numbers we're calculating
    # against, then flatten the array to a vector
    q = nm.ravel(xx+yy[:, nm.newaxis]).astype(nm.complex128)
    # create z as a 0+0j array of the same length as q
    z = nm.zeros(q.shape, nm.complex128)

    start_time = datetime.datetime.now()
    print "Total elements:", len(q)
    output = calculate_z_gpu_elementwise(q, maxiter, z)
    end_time = datetime.datetime.now()
    secs = end_time - start_time
    print "Main took", secs

    validation_sum = sum(output)
    print "Total sum of elements (for validation):", validation_sum

    if show_output: 
        import Image
        output = (output + (256*output) + (256**2)*output) * 8
        im = Image.new("RGB", (w/2, h/2))
        im.fromstring(output.tostring(), "raw", "RGBX", 0, -1)
        im.show()


if __name__ == '__main__':
    w = int(sys.argv[1]) # e.g. 100
    h = int(sys.argv[1]) # e.g. 100
    maxiter = int(sys.argv[2]) # e.g. 300

    calculate(True)



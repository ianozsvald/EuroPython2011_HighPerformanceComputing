# Use to show the speed difference between CPU and GPU calculations
# ian@ianozsvald.com July 2011

import sys
import datetime
import numpy as np


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
    complex_type = np.complex64 # or nm.complex128 on newer CUDA devices
    #float_type = np.float32 # or nm.float64 on newer CUDA devices
    output = np.resize(np.array(0,), q.shape)
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
    # make a list of x and y values which will represent q
    # xx and yy are the co-ordinates, for the default configuration they'll look like:
    # if we have a 1000x1000 plot
    # xx = [-2.13, -2.1242, -2.1184000000000003, ..., 0.7526000000000064, 0.7584000000000064, 0.7642000000000064]
    # yy = [1.3, 1.2948, 1.2895999999999999, ..., -1.2844000000000058, -1.2896000000000059, -1.294800000000006]
    x_step = (float(x2 - x1) / float(w)) * 2
    y_step = (float(y1 - y2) / float(h)) * 2
    x=[]
    y=[]
    ycoord = y2
    while ycoord > y1:
        y.append(ycoord)
        ycoord += y_step
    xcoord = x1
    while xcoord < x2:
        x.append(xcoord)
        xcoord += x_step
    
    x = np.array(x)
    y = np.array(y) * 1j # make y a complex number
    print "x and y have length:", len(x), len(y)

    # create a square matrix using clever addressing
    x_y_square_matrix = x+y[:, np.newaxis] # it is np.complex128
    # convert square matrix to a flatted vector using ravel
    q = np.ravel(x_y_square_matrix)
    # create z as a 0+0j array of the same length as q
    # note that it defaults to reals (float64) unless told otherwise
    z = np.zeros(q.shape, np.complex128)


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



# Mandelbrot calculate using GPU, Serial numpy and faster numpy
# Use to show the speed difference between CPU and GPU calculations
# ian@ianozsvald.com July 2011

# Originally based on vegaseat's TKinter/numpy example code from 2006
# http://www.daniweb.com/code/snippet216851.html#
# with minor changes to move to numpy from the obsolete Numeric

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

def calculate_z_asnumpy_gpu(q, maxiter, z):
    """Calculate z using numpy on the GPU"""
    # convert complex128s (2*float64) to complex64 (2*float32) so they run
    # on older CUDA cards like the one in my MacBook. To use float64 doubles
    # just edit these two lines
    complex_type = np.complex64 # or nm.complex128 on newer CUDA devices
    float_type = np.float32 # or nm.float64 on newer CUDA devices

    # create an output array on the gpu of int32 as one long vector
    outputg = gpuarray.to_gpu(np.resize(np.array(0,), q.shape))
    # resize our z and g as necessary to longer or shorter float types
    z = z.astype(complex_type)
    q = q.astype(complex_type)
    # create zg and qg on the gpu
    zg = gpuarray.to_gpu(z)
    qg = gpuarray.to_gpu(q)
    # create 2.0 as an array
    twosg = gpuarray.to_gpu(np.array([2.0]*zg.size).astype(float_type))
    # create 0+0j as an array
    cmplx0sg = gpuarray.to_gpu(np.array([0+0j]*zg.size).astype(complex_type))
    # create a bool array to hold the (for abs_zg > twosg) result later
    comparison_result = gpuarray.to_gpu(np.array([False]*zg.size).astype(np.bool))
    # we'll add 1 to iterg after each iteration, create an array to hold the iteration count
    iterg = gpuarray.to_gpu(np.array([0]*zg.size).astype(np.int32))
    
    for iter in range(maxiter):
        # multiply z on the gpu by itself, add q (on the gpu)
        zg = zg*zg + qg
        # abs returns a complex (rather than a float) from the complex
        # input where the real component is the absolute value (which
        # looks like a bug) so I take the .real after abs()
        # the above bug relates to pyCUDA from mid2010, it might be fixed now...
        abs_zg = abs(zg).real
       
        # figure out if zg is > 2
        comparison_result = abs_zg > twosg
        # based on the result either take 0+0j for qg and zg or leave unchanged
        qg = gpuarray.if_positive(comparison_result, cmplx0sg, qg)
        zg = gpuarray.if_positive(comparison_result, cmplx0sg, zg)
        # if the comparison is true then update the iterations count to outputg
        # which we'll extract later
        outputg = gpuarray.if_positive(comparison_result, iterg, outputg)
        # increment the iteration counter
        iterg = iterg + 1
    # extract the result from the gpu back to the cpu
    output = outputg.get()
    return output


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
    output = calculate_z_asnumpy_gpu(q, maxiter, z)
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



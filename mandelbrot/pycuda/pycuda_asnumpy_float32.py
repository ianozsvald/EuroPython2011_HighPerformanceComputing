# Mandelbrot calculate using GPU, Serial numpy and faster numpy
# Use to show the speed difference between CPU and GPU calculations
# ian@ianozsvald.com July 2011

# Originally based on vegaseat's TKinter/numpy example code from 2006
# http://www.daniweb.com/code/snippet216851.html#
# with minor changes to move to numpy from the obsolete Numeric

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

def calculate_z_asnumpy_gpu(q, maxiter, z):
    """Calculate z using numpy on the GPU"""
    # convert complex128s (2*float64) to complex64 (2*float32) so they run
    # on older CUDA cards like the one in my MacBook. To use float64 doubles
    # just edit these two lines
    complex_type = nm.complex64 # or nm.complex128 on newer CUDA devices
    float_type = nm.float32 # or nm.float64 on newer CUDA devices

    # create an output array on the gpu of int32 as one long vector
    outputg = gpuarray.to_gpu(nm.resize(nm.array(0,), q.shape))
    # resize our z and g as necessary to longer or shorter float types
    z = z.astype(complex_type)
    q = q.astype(complex_type)
    # create zg and qg on the gpu
    zg = gpuarray.to_gpu(z)
    qg = gpuarray.to_gpu(q)
    # create 2.0 as an array
    twosg = gpuarray.to_gpu(nm.array([2.0]*zg.size).astype(float_type))
    # create 0+0j as an array
    cmplx0sg = gpuarray.to_gpu(nm.array([0+0j]*zg.size).astype(complex_type))
    # create a bool array to hold the (for abs_zg > twosg) result later
    comparison_result = gpuarray.to_gpu(nm.array([False]*zg.size).astype(nm.bool))
    # we'll add 1 to iterg after each iteration, create an array to hold the iteration count
    iterg = gpuarray.to_gpu(nm.array([0]*zg.size).astype(nm.int32))
    
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



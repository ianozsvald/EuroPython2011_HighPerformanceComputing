# Mandelbrot calculate using GPU, Serial numpy and faster numpy
# Use to show the speed difference between CPU and GPU calculations
# ian@ianozsvald.com March 2010

# Based on vegaseat's TKinter/numpy example code from 2006
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




complex_gpu_sm_newindexing = SourceModule("""
        // original newindexing code using original mandelbrot pycuda
        #include <pycuda-complex.hpp>

        __global__ void calc_gpu_sm_insteps(pycuda::complex<float> *z, pycuda::complex<float> *q, int *iteration, int maxiter, const int nbritems) {
            //const int i = blockDim.x * blockIdx.x + threadIdx.x;
            unsigned tid = threadIdx.x;
            unsigned total_threads = gridDim.x * blockDim.x;
            unsigned cta_start = blockDim.x * blockIdx.x;

            for ( int i = cta_start + tid; i < nbritems; i += total_threads) {
                for (int n=0; n < maxiter; n++) {
                    z[i] = (z[i]*z[i])+q[i]; 
                    if (abs(z[i]) > 2.0f) {
                        iteration[i]=n; 
                        z[i] = pycuda::complex<float>(); 
                        q[i] = pycuda::complex<float>();
                    }
                };            
            }
        }
        """)


calc_gpu_sm_newindexing = complex_gpu_sm_newindexing.get_function('calc_gpu_sm_insteps')
print 'complex_gpu_sm:'
print 'Registers', calc_gpu_sm_newindexing.num_regs
print 'Local mem', calc_gpu_sm_newindexing.local_size_bytes, 'bytes'
print 'Shared mem', calc_gpu_sm_newindexing.shared_size_bytes, 'bytes'





complex_gpu_sm = SourceModule("""
        #include <pycuda-complex.hpp>

        __global__ void calc_gpu_sm(pycuda::complex<float> *z, pycuda::complex<float> *q, int *iteration, int maxiter) {
            const int i = blockDim.x * blockIdx.x + threadIdx.x;
            iteration[i] = 0;
            for (int n=0; n < maxiter; n++) {
                z[i] = (z[i]*z[i])+q[i]; 
                if (abs(z[i]) > 2.0f) {
                    iteration[i]=n; 
                    z[i] = pycuda::complex<float>(); 
                    q[i] = pycuda::complex<float>();
                }
                   
                //iteration[i] = abs(q[i]);
            };            
        }
        """, keep=True)

calc_gpu_sm = complex_gpu_sm.get_function('calc_gpu_sm')
print 'complex_gpu_sm:'
print 'Registers', calc_gpu_sm.num_regs
print 'Local mem', calc_gpu_sm.local_size_bytes, 'bytes'
print 'Shared mem', calc_gpu_sm.shared_size_bytes, 'bytes'





def calculate_z_gpu_sourcemodule(q, maxiter, z):
    z = z.astype(nm.complex64)
    q = q.astype(nm.complex64)
    output = nm.resize(nm.array(0,), q.shape)
    # calc_gpu_sm is limited in size to whatever's the max GridX size
    #calc_gpu_sm(drv.In(z), drv.In(q), drv.Out(output), numpy.int32(maxiter), grid=(len(q),1), block=(1,1,1))
    # calc_gpu_sm_newindexing uses a step to iterate through larger amounts of data
    calc_gpu_sm_newindexing(drv.In(z), drv.In(q), drv.InOut(output), numpy.int32(maxiter), numpy.int32(len(q)), grid=(400,1), block=(512,1,1))


    #calc_gpu_sm_lightweight(drv.In(z), drv.In(q), drv.Out(output), numpy.int32(maxiter), grid=(blocks,1), block=(1,1,1))
    #calc_gpu_sm_newindexing(drv.In(z), drv.In(q), drv.InOut(output), numpy.int32(maxiter), numpy.int32(len(q)), grid=(480,1), block=(48,1,1))

    # double precision
    #z = z.astype(nm.complex128)
    #q = q.astype(nm.complex128)
    #output = output.astype(nm.complex128)
    #calc_gpu_sm_newindexing_double(drv.In(z), drv.In(q), drv.InOut(output), numpy.int32(maxiter), numpy.int32(len(q)), grid=(480,1), block=(512,1,1))

    return output

complex_gpu = ElementwiseKernel(
        """pycuda::complex<float> *z, pycuda::complex<float> *q, int *iteration, int maxiter""",
            """for (int n=0; n < maxiter; n++) {z[i] = (z[i]*z[i])+q[i]; if (abs(z[i]) > 2.00f) {iteration[i]=n; z[i] = pycuda::complex<float>(); q[i] = pycuda::complex<float>();};};""",
        "complex5",
        preamble="""#include <pycuda-complex.hpp>""",
        keep=True)


def calculate_z_gpu_elementwise(q, maxiter, z):
    output = nm.resize(nm.array(0,), q.shape)
    q_gpu = gpuarray.to_gpu(q.astype(nm.complex64))
    z_gpu = gpuarray.to_gpu(z.astype(nm.complex64))
    iterations_gpu = gpuarray.to_gpu(output) 
    print "maxiter gpu", maxiter
    # the for loop and complex calculations are all done on the GPU
    # we bring the iterations_gpu array back to determine pixel colours later
    complex_gpu(z_gpu, q_gpu, iterations_gpu, maxiter)

    iterations = iterations_gpu.get()
    return iterations


def calculate_z_asnumpy_gpu(q, maxiter, z):
    """Calculate z using numpy on the GPU"""
    outputg = gpuarray.to_gpu(nm.resize(nm.array(0,), q.shape))
    zg = gpuarray.to_gpu(z.astype(nm.complex64))
    qg = gpuarray.to_gpu(q.astype(nm.complex64))
    # 2.0 as an array
    twosg = gpuarray.to_gpu(nm.array([2.0]*zg.size).astype(numpy.float32))
    # 0+0j as an array
    cmplx0sg = gpuarray.to_gpu(nm.array([0+0j]*zg.size).astype(nm.complex64))
    # for abs_zg > twosg result
    comparison_result = gpuarray.to_gpu(nm.array([False]*zg.size).astype(nm.bool))
    # we'll add 1 to iterg after each iteration
    iterg = gpuarray.to_gpu(nm.array([0]*zg.size).astype(nm.int32))
    
    for iter in range(maxiter):
        zg = zg*zg + qg

        # abs returns a complex (rather than a float) from the complex
        # input where the real component is the absolute value (which
        # looks like a bug) so I take the .real after abs()
        abs_zg = abs(zg).real
       
        comparison_result = abs_zg > twosg
        qg = gpuarray.if_positive(comparison_result, cmplx0sg, qg)
        zg = gpuarray.if_positive(comparison_result, cmplx0sg, zg)
        outputg = gpuarray.if_positive(comparison_result, iterg, outputg)
        iterg = iterg + 1
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
    #output = calculate_z_asnumpy_gpu(q, maxiter, z)
    #output = calculate_z_gpu_elementwise(q, maxiter, z)
    output = calculate_z_gpu_sourcemodule(q, maxiter, z)
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



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

# This version can step through arbitrarily-large arrays
# Note this is simply based on the ElementwiseKernel's internal code
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


# This version is restricted to stepping through arrays only as big as the Grid will allow
# i.e. it lacks proper indexing
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
    complex_type = nm.complex64 # or nm.complex128 on newer CUDA devices
    #float_type = nm.float32 # or nm.float64 on newer CUDA devices
    z = z.astype(complex_type)
    q = q.astype(complex_type)
    output = nm.resize(nm.array(0,), q.shape)
    # calc_gpu_sm is limited in size to whatever's the max GridX size (i.e. probably can't do 1000x1000 grids!)
    #calc_gpu_sm(drv.In(z), drv.In(q), drv.Out(output), numpy.int32(maxiter), grid=(len(q),1), block=(1,1,1))
    
    # calc_gpu_sm_newindexing uses a step to iterate through larger amounts of data (i.e. can do 1000x1000 grids!)
    calc_gpu_sm_newindexing(drv.In(z), drv.In(q), drv.InOut(output), numpy.int32(maxiter), numpy.int32(len(q)), grid=(400,1), block=(512,1,1))

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



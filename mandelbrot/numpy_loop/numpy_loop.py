# Mandelbrot calculate using GPU, Serial numpy and faster numpy
# Use to show the speed difference between CPU and GPU calculations
# ian@ianozsvald.com March 2010

# Based on vegaseat's TKinter/numpy example code from 2006
# http://www.daniweb.com/code/snippet216851.html#
# with minor changes to move to numpy from the obsolete Numeric

import datetime
import sys
import numpy as np
import Image

# You can choose a calculation routine below (calculate_z), uncomment
# one of the three lines to test the three variations
# Speed notes are listed in the same place

# area of space to investigate
x1, x2, y1, y2 = -2.13, 0.77, -1.3, 1.3


def calculate_z(xs, ys, maxiter):
    """ Generate a mandelbrot set """
    N = len(xs)
    M = len(ys)
    
    d = np.zeros((M, N)).astype(np.int)
    for j in range(M):
        for i in range(N):
            q = xs[i] + ys[j]*1j
            z = 0+0j
            for iteration in range(maxiter):
                # straight math is faster!
                z = z*z + q
                # breaking the math out makes it run at half the speed!
                #if z.real*z.real + z.imag*z.imag > 4.0:
                if abs(z) > 2:
                   break
            else:
                iteration = 0 
            d[j,i] = iteration
    return d


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

    #import ipdb; ipdb.set_trace()

    print "Total elements:", len(y)*len(x)
    start_time = datetime.datetime.now()
    output = calculate_z(x, y, maxiter)
    end_time = datetime.datetime.now()
    secs = end_time - start_time
    print "Main took", secs

    validation_sum = np.sum(output)
    print "Total sum of elements (for validation):", validation_sum

    if show_output: 
        output = (output + (256*output) + (256**2)*output) * 8
        im = Image.new("RGB", (w/2, h/2))
        im.fromstring(output.tostring(), "raw", "RGBX", 0, -1)
        #im.save('mandelbrot.png')
        im.show()

# test the class
if __name__ == '__main__':
    w = int(sys.argv[1]) # e.g. 100
    h = int(sys.argv[1]) # e.g. 100
    maxiter = int(sys.argv[2]) # e.g. 300

    calculate(True)

    

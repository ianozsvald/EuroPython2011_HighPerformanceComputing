# Mandelbrot calculate using GPU, Serial numpy and faster numpy
# Use to show the speed difference between CPU and GPU calculations
# ian@ianozsvald.com March 2010

# Based on vegaseat's TKinter/numpy example code from 2006
# http://www.daniweb.com/code/snippet216851.html#
# with minor changes to move to numpy from the obsolete Numeric

import datetime
import sys
import numpy as nm
import Image

import calculate_z

# You can choose a calculation routine below (calculate_z), uncomment
# one of the three lines to test the three variations
# Speed notes are listed in the same place

# area of space to investigate
x1, x2, y1, y2 = -2.13, 0.77, -1.3, 1.3


def calculate(show_output):
    # make a list of x and y values
    # xx is e.g. -2.13,...,0.712
    xx = nm.arange(x1, x2, (x2-x1)/w*2)
    # yy is e.g. 1.29,...,-1.24
    yy = nm.arange(y2, y1, (y1-y2)/h*2) 
    # we see a rounding error for arange on yy with h==1000
    # so here I correct for it
    if len(yy) > h / 2.0:
        yy = yy[:-1]
    assert len(xx) == w / 2.0
    assert len(yy) == h / 2.0

    print "Total elements:", len(yy)*len(xx)
    start_time = datetime.datetime.now()
    output = calculate_z.calculate_z(xx, yy, maxiter)
    end_time = datetime.datetime.now()
    secs = end_time - start_time
    print "Main took", secs

    validation_sum = nm.sum(output)
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

    

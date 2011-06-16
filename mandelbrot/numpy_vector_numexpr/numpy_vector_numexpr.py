import datetime
import sys
import numpy as nm
import numexpr

# area of space to investigate
x1, x2, y1, y2 = -2.13, 0.77, -1.3, 1.3

# use numexpr library to vectorise (and maybe parallelise) the numpy expressions

def calculate_z_numpy(q, maxiter, z):
    output = nm.resize(nm.array(0,), q.shape)
    for iteration in range(maxiter):
        #z = z*z + q
        z = numexpr.evaluate("z*z+q")
        #done = nm.greater(abs(z), 2.0)
        done = numexpr.evaluate("abs(z).real>2.0")
        #q = nm.where(done,0+0j, q)
        q = numexpr.evaluate("where(done, 0+0j, q)")
        #z = nm.where(done,0+0j, z)
        z = numexpr.evaluate("where(done,0+0j, z)")
        #output = nm.where(done, iteration, output)
        output = numexpr.evaluate("where(done, iteration, output)")
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
    output = calculate_z_numpy(q, maxiter, z)
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

    

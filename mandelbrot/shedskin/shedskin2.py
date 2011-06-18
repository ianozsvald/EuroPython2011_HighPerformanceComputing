import sys
import datetime
import math
# area of space to investigate
x1, x2, y1, y2 = -2.13, 0.77, -1.3, 1.3

# Mark Darfour's version without complex numbers in critical loop

def calculate_z_serial_purepython(q, maxiter, z):
    output = [0] * len(q)
    for i in range(len(q)):
        zx, zy = z[i].real, z[i].imag
        qx, qy = q[i].real, q[i].imag
        for iteration in range(maxiter):
            # expand complex numbers to floats, do raw float arithmetic
            # as the shedskin variant isn't so fast
            # I believe MD said that complex numbers are allocated on the heap
            # and this could easily be improved for the next shedskin
            zx_new = (zx * zx - zy * zy) + qx
            zy_new = (2 * (zx * zy)) + qy # note that zx(old) is used so we make zx_new on previous line
            zx = zx_new
            zy = zy_new
            # remove need for abs and just square the numbers
            if zx*zx + zy*zy > 4.0:
                output[i] = iteration
                break
    return output

def calc_pure_python(show_output):
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
    q = []
    for ycoord in y:
        for xcoord in x:
            q.append(complex(xcoord,ycoord))
    z = [0+0j] * len(q)

    print "Total elements:", len(z)    
    start_time = datetime.datetime.now()
    output = calculate_z_serial_purepython(q, maxiter, z)
    end_time = datetime.datetime.now()
    secs = end_time - start_time
    print "Main took", secs

    validation_sum = sum(output)
    print "Total sum of elements (for validation):", validation_sum

    # uncomment this to verify image output, use Python only
    #if show_output: 
    #    import Image
    #    import numpy as nm
    #    output = nm.array(output)
    #    output = (output + (256*output) + (256**2)*output) * 8
    #    im = Image.new("RGB", (w/2, h/2))
    #    # you can experiment with these x and y ranges
    #    im.fromstring(output.tostring(), "raw", "RGBX", 0, -1)
    #    #im.save('mandelbrot.png')
    #    im.show()

if __name__ == "__main__":
    # get width, height and max iterations from cmd line
    # 'python mandelbrot_shedskin2.py 1000 300'
    # 'pypy mandelbrot_shedskin2.py 1000 300'
    # './mandelbrot2 1000 300'
    w = int(sys.argv[1]) # e.g. 100
    h = int(sys.argv[1]) # e.g. 100
    maxiter = int(sys.argv[2]) # e.g. 300
    
    # only set show_output to True if we're running through Python
    calc_pure_python(False)

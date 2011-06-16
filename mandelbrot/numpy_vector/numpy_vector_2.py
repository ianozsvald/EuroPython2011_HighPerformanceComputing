import datetime
import sys
import numpy as np

# area of space to investigate
x1, x2, y1, y2 = -2.13, 0.77, -1.3, 1.3

# we use a STEP_SIZE to jump through the data in blocks to try to get 
# better use out of L2 cache (mixed success)
# an early stopping criteria is used, this works correctly but doesn't seem to help
# POSSIBLE - set output to -1, count nbr of -1s left, set -1s at end to 0?

def calculate_z_numpy(q_full, maxiter, z_full):
    output = np.resize(np.array(0,), q_full.shape)
    #STEP_SIZE = len(q_full) # 54s for 250,000
    #STEP_SIZE = 90000 # 52
    #STEP_SIZE = 50000 # 45s
    #STEP_SIZE = 45000 # 45s
    #STEP_SIZE = 20000 # 42s
    #STEP_SIZE = 10000 # 43s
    #STEP_SIZE = 5000 # 45s
    #STEP_SIZE = 1000 # 1min02
    STEP_SIZE = 100
    print "STEP_SIZE", STEP_SIZE
    for step in range(0, len(q_full), STEP_SIZE):
        z = z_full[step:step+STEP_SIZE]
        q = q_full[step:step+STEP_SIZE]
        #nbr_done = 0
        for iteration in range(maxiter):
            z = z*z + q
            done = np.greater(abs(z), 2.0)
            q = np.where(done,0+0j, q)
            z = np.where(done,0+0j, z)
            output[step:step+STEP_SIZE] = np.where(done, iteration, output[step:step+STEP_SIZE])
            #nbr_just_done = np.sum(np.where(done, 1, 0))
            #nbr_done += nbr_just_done
            #if (output[step:step+STEP_SIZE] > 0).all():
            #if nbr_done == STEP_SIZE:
            #    break
    return output


def calculate(show_output):
    # make a list of x and y values
    # xx is e.g. -2.13,...,0.712
    xx = np.arange(x1, x2, (x2-x1)/w*2)
    # yy is e.g. 1.29,...,-1.24
    yy = np.arange(y2, y1, (y1-y2)/h*2) * 1j
    # we see a rounding error for arange on yy with h==1000
    # so here I correct for it
    if len(yy) > h / 2.0:
        yy = yy[:-1]
    assert len(xx) == w / 2.0
    assert len(yy) == h / 2.0

    print "xx and yy have length", len(xx), len(yy)

    # yy will become 0+yyj when cast to complex128 (2 * 8byte float64) same as Python float 
    yy = yy.astype(np.complex128)
    # create q as a square matrix initially of complex numbers we're calculating
    # against, then flatten the array to a vector
    q = np.ravel(xx+yy[:, np.newaxis]).astype(np.complex128)
    # create z as a 0+0j array of the same length as q
    z = np.zeros(q.shape, np.complex128)

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

    

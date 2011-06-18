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
    STEP_SIZE = 20000 # 42s # roughly this looks optimal on Macbook and dual core desktop i3
    #STEP_SIZE = 10000 # 43s
    #STEP_SIZE = 5000 # 45s
    #STEP_SIZE = 1000 # 1min02
    #STEP_SIZE = 100 # 3mins
    print "STEP_SIZE", STEP_SIZE
    for step in range(0, len(q_full), STEP_SIZE):
        z = z_full[step:step+STEP_SIZE]
        q = q_full[step:step+STEP_SIZE]
        for iteration in range(maxiter):
            z = z*z + q
            done = np.greater(abs(z), 2.0)
            q = np.where(done,0+0j, q)
            z = np.where(done,0+0j, z)
            output[step:step+STEP_SIZE] = np.where(done, iteration, output[step:step+STEP_SIZE])
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

    

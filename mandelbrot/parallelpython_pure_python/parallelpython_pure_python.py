import itertools
import multiprocessing
import sys
import datetime
import pp

# area of space to investigate
x1, x2, y1, y2 = -2.13, 0.77, -1.3, 1.3


def calculate_z_serial_purepython(chunk):
    """Take a tuple of (q, maxiter, z), create an output array of iterations for Mandelbrot set"""
    q, maxiter, z = chunk
    output = [0] * len(q)
    for i in range(len(q)):
        zi = z[i]
        qi = q[i]
        for iteration in range(maxiter):
            zi = zi * zi + qi
            if abs(zi) > 2.0:
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

    # split work list into contiguous chunks, one per CPU
    # build this into chunks which we'll apply to map_async
    nbr_chunks = multiprocessing.cpu_count()
    chunk_size = len(q) / nbr_chunks

    # split our long work list into smaller chunks
    # make sure we handle the edge case where nbr_chunks doesn't evenly fit into len(q)
    import math
    if len(q) % nbr_chunks != 0:
        # make sure we get the last few items of data when we have
        # an odd size to chunks (e.g. len(q) == 100 and nbr_chunks == 3
        nbr_chunks += 1
    chunks = [(q[x*chunk_size:(x+1)*chunk_size],maxiter,z[x*chunk_size:(x+1)*chunk_size]) for x in xrange(nbr_chunks)]
    print chunk_size, len(chunks), len(chunks[0][0])


    start_time = datetime.datetime.now()

    # tuple of all parallel python servers to connect with
    ppservers = () # use this machine
    # I can't get autodiscover to work at home
    #ppservers=("*",) # autodiscover on network

    job_server = pp.Server(ppservers=ppservers)
    # it'll autodiscover the nbr of cpus it can use if first arg not specified

    print "Starting pp with", job_server.get_ncpus(), "local CPU workers"
    output = []
    jobs = []
    for chunk in chunks:
        print "Submitting job with len(q) {}, len(z) {}".format(len(chunk[0]), len(chunk[2]))
        job = job_server.submit(calculate_z_serial_purepython, (chunk,), (), ())
        jobs.append(job)
    for job in jobs:
        output_job = job()
        output += output_job
    # print statistics about the run
    print job_server.print_stats()

    end_time = datetime.datetime.now()

    secs = end_time - start_time
    print "Main took", secs

    validation_sum = sum(output)
    print "Total sum of elements (for validation):", validation_sum

    if show_output: 
        import Image
        import numpy as nm
        output = nm.array(output)
        output = (output + (256*output) + (256**2)*output) * 8
        im = Image.new("RGB", (w/2, h/2))
        # you can experiment with these x and y ranges
        im.fromstring(output.tostring(), "raw", "RGBX", 0, -1)
        #im.save('mandelbrot.png')
        im.show()

if __name__ == "__main__":
    # get width, height and max iterations from cmd line
    # 'python mandelbrot_pypy.py 100 300'
    w = int(sys.argv[1]) # e.g. 100
    h = int(sys.argv[1]) # e.g. 100
    maxiter = int(sys.argv[2]) # e.g. 300
    
    # we can show_output for Python, not for PyPy
    calc_pure_python(True)

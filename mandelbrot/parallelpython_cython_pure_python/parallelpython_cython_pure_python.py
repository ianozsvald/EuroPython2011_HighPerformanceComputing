import itertools
import multiprocessing
import sys
import datetime
import pp

# area of space to investigate
x1, x2, y1, y2 = -2.13, 0.77, -1.3, 1.3

# assume that calculate_z is in *local* directory
import calculate_z


def calc(inps):
    """use the calculate_z module's calculate_z to process
       the tuple of inps"""
    q, maxiter, z = inps
    return calculate_z.calculate_z(q, maxiter, z)


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

    # split work list into continguous chunks, one per CPU
    #chunk_size = len(q) / multiprocessing.cpu_count()
    chunk_size = len(q) / 16
    chunks = []
    chunk_number = 0
    while True:
        chunk = (q[chunk_number * chunk_size:chunk_size*(chunk_number+1)], maxiter, z[chunk_number * chunk_size:chunk_size*(chunk_number+1)]) 
        chunks.append(chunk)
        chunk_number += 1
        if chunk_size * chunk_number > len(q):
            break
    print chunk_size, len(chunks), len(chunks[0][0])

    start_time = datetime.datetime.now()

    ppservers = ('localhost',)
    # we MUST start 'ppserver.py -d' in this same directory in another
    # term or else it can't see calculate_z.so and it'll barf!
    # i.e. module must be in search path somehow

    NBR_LOCAL_CPUS = 0 # if 0, it sends jobs out to other ppservers
    job_server = pp.Server(NBR_LOCAL_CPUS, ppservers=ppservers)

    print "Starting pp with", job_server.get_ncpus(), "workers"
    output = []
    jobs = []
    for chunk in chunks:
        # specify with last tuple item that pp should import calculate_z
        # module - this means it must be in the PYTHONPATH for ppserver.py
        # or the local job_server
        job = job_server.submit(calc, (chunk,), (), ("calculate_z",))
        assert job is not None
        jobs.append(job)
    for job in jobs:
        output_job = job()
        assert output_job is not None
        output += output_job
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

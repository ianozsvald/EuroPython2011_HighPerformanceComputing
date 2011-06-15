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
    return calculate_z.calculate_z(inps)


def calc_pure_python(show_output):
    x_step = (float(x2 - x1) / float(w)) * 2
    y = y2
    y_step = (float(y1 - y2) / float(h)) * 2
    q = []
    while y > y1:
        x = x1
        while x < x2:
            q.append(complex(x,y))
            x += x_step
        y += y_step
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

    NBR_LOCAL_CPUS = 0 # if 0, it sends jobs out to other ppservers
    job_server = pp.Server(NBR_LOCAL_CPUS, ppservers=ppservers)
    # it'll autodiscover the nbr of cpus it can use

    print "Starting pp with", job_server.get_ncpus(), "workers"
    output = []
    jobs = []
    for chunk in chunks:
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

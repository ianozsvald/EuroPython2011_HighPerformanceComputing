# based on calculate_z_serial_purepython
def calculate_z(inps):
    cdef unsigned int i
    cdef int maxiter, iteration
    cdef complex zi, qi
    cdef list q, output

    q, maxiter, z = inps
    output = [0] * len(q)
    for i in range(len(q)):
        zi = complex() #z[i]
        qi = q[i]
        for iteration in range(maxiter):
            zi = zi * zi + qi
            if abs(zi) > 2.0:
                output[i] = iteration
                break
    return output

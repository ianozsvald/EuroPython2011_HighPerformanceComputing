# based on calculate_z_serial_purepython
def calculate_z(list q, int maxiter, list z):
    cdef unsigned int i
    cdef int iteration
    cdef complex zi, qi
    cdef list output

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

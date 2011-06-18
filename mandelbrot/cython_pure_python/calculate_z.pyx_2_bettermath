def calculate_z(list q, int maxiter, list z):
    cdef unsigned int i
    cdef int iteration
    cdef list output
    cdef double zx, zy, qx, qy, zx_new, zy_new

    output = [0] * len(q)
    for i in range(len(q)):
        zx = z[i].real # need to extract items using dot notation
        zy = z[i].imag
        qx = q[i].real
        qy = q[i].imag

        for iteration in range(maxiter):
            zx_new = (zx * zx - zy * zy) + qx
            zy_new = (zx * zy + zy * zx) + qy
            # must assign after else we're using the new zx/zy in the fla
            zx = zx_new
            zy = zy_new
            # note - math.sqrt makes this almost twice as slow!
            #if math.sqrt(zx*zx + zy*zy) > 2.0:
            if (zx*zx + zy*zy) > 4.0:
                output[i] = iteration
                break
    return output


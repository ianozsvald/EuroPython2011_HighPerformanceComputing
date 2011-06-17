
# based on calculate_z_serial_purepython
def calculate_z(list q, int maxiter, list z):
    cdef unsigned int i
    #cdef int maxiter, 
    cdef int iteration
    cdef complex zi, qi
    #cdef list q
    cdef list output

    #q, maxiter, z = inps
    output = [0] * len(q)
    for i in range(len(q)):
        #if i % 1000 == 0:
        #    # print out some progress info since it is so slow...
        #    print "%0.2f%% complete" % (1.0/len(q) * i * 100)
        zi = complex() #z[i]
        qi = q[i]
        for iteration in range(maxiter):
            #z[i] = z[i]*z[i] + q[i]
            zi = zi * zi + qi
            #if abs(z[i]) > 2.0:
            if abs(zi) > 2.0:
                #q[i] = 0+0j
                #z[i] = 0+0j
                output[i] = iteration
                break
    return output



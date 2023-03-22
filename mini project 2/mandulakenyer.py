import numpy as np
from numba import jit
import multiprocessing as mp
import dask.array as da

# re_prop : list of properties
# re_prop[0] : re_min, re_prop[1] : re_max, re_prop[2] : re_scale
# im : same as re
# dt : data type
def create_imre(re_prop, im_prop, dt=None):
    re = np.linspace(re_prop[0], re_prop[1], re_prop[2], dtype=dt)
    im = np.linspace(im_prop[0], im_prop[1], im_prop[2],  dtype=dt) *1j
    
    return re, im

# naive 
def naive(re, im, dt, itr, thresh):
    re_val, im_val = create_imre(re, im, dt)
    M = np.zeros((re_val.size, im_val.size), dtype=dt)
    
    for a in range(re[2]): 
        for b in range(im[2]):
            z = 0 + 0j
            c = re_val[a] + im_val[b]

            for i in range(itr):
                z = z**2 + c
            
                if abs(z) > thresh:
                    M[b, a] = i
                    break
            else:
                M[b, a] = itr

    return M

# numpy
def vectorized(re, im, dt, comp_dt, itr, thresh):
    re_val, im_val = create_imre(re, im, dt)
    M = np.zeros((re_val.size, im_val.size), dtype=dt)
    z = np.zeros((re_val.size, im_val.size), dtype=comp_dt)
    c = re_val + im_val[:, np.newaxis]

    for i in range(itr):
        z = z**2 + c
        M[thresh <= abs(z)] = i

    M[M == 0] = itr

    return M

# multiprocessing
def parallel_mb(c_matrix, itr, thresh, dt=None):
    M = np.zeros(c_matrix.shape[0], dtype=dt)
    
    for count, value in enumerate(c_matrix):
        c = value
        z = 0 + 0j
        
        for i in range(itr):
            z = z**2 + c
            
            if abs(z) > thresh:
                M[count] = i 
                break
        else:
            M[count] = itr
            
    
    return M

def parallelize(func, proc_num, chunk_num, re, im, dt, itr, thresh):
    re_val, im_val = create_imre(re, im, dt)
    comp_matrix = re_val + im_val[:, np.newaxis]
    items = [(C, itr, thresh, dt) for C in comp_matrix]

    pool = mp.Pool(processes=proc_num)
    output = [pool.starmap(func, items, chunksize=chunk_num)]
    
    pool.close()
    pool.join()
    
    return output

# dask implementation
def dask_mb(c_matrix, chunk_size, itr, thresh):
    M = da.zeros(c_matrix.shape, chunks=(chunk_size,chunk_size))
    z = da.zeros(c_matrix.shape, chunks=(chunk_size,chunk_size))

    for i in range(itr):
        z = z**2 + c_matrix
        M[da.abs(z) < thresh] = i
    
    return M

def dask_parallel(func, c_matrix, chunk_size, itr, thresh):
    
    output = da.map_blocks(func, c_matrix, chunk_size, itr, thresh)
    output = output.compute()
    
    return output
'''
'mandulakenyer' module containts different implementations of Mandelbrot algorithm.
'''

import numpy as np
import multiprocessing as mp

def create_imre(re_prop, im_prop):
    '''
    Returns 2 arrays 

    Inputs
        re_prop : list of properties of on real-axis - [minimum, maximum, scale]
        im_prop : list of properties of on imaginary-axis - [minimum, maximum, scale]
    
    Outputs
        re :
        im : 
    '''

    re = np.linspace(re_prop[0], re_prop[1], re_prop[2])
    im = np.linspace(im_prop[0], im_prop[1], im_prop[2]) *1j
    
    return re, im


def naive(re, im, itr, thresh):
    '''
    Implements a naive version of Mandelbrot algorithm
    Returns a Mandelbrot dataset.

    Inputs
        re : list of properties of on real-axis - [minimum, maximum, scale]
        im : list of properties of on imaginary-axis - [minimum, maximum, scale]
        itr : iteration number
        thresh : threshold
    
    Outputs
        M : Mandelbrot dataset
    '''

    re_val, im_val = create_imre(re, im)
    M = np.zeros((re_val.size, im_val.size))
    
    for a in range(re[2]): 
        for b in range(im[2]):
            z = 0 + 0j
            c = re_val[a] + im_val[b]

            for i in range(itr):
                z = z**2 + c
            
                if abs(z) > thresh:
                    M[a, b] = i
                    break
            else:
                M[a, b] = itr

    return M

def vectorized(re, im, itr, thresh):
    '''
    Implements a vectorized version of Mandelbrot algorithm
    Returns a Mandelbrot dataset.

    Inputs
        re : 
        im : 
        itr : uteration number
        thresh : threshold
    
    Outputs
        M : Mandelbrot dataset
    '''

    re_val, im_val = create_imre(re, im)
    M = np.zeros((re_val.size, im_val.size))
    z = np.zeros((re_val.size, im_val.size))
    c = re_val + im_val[:, np.newaxis]

    for i in range(itr):
        z = z**2 + c
        M[thresh <= abs(z)] = i

    M[M == 0] = itr

    return M



def parallel_mb(c_matrix, itr, thresh):
    '''
    TODO : maybe later
    Helper function for parallelized implementation
    '''
    M = np.zeros(c_matrix.shape[0])
    
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

def parallelize(func, proc_num, chunk_num, re, im, itr, thresh):
    '''
    TODO : maybe later
    Parallelized implementation of Mandelbrot
    '''

    re_val, im_val = create_imre(re, im)
    comp_matrix = re_val + im_val[:, np.newaxis]
    items = [(C, itr, thresh) for C in comp_matrix]

    pool = mp.Pool(processes=proc_num)
    output = [pool.starmap(func, items, chunksize=chunk_num)]
    
    pool.close()
    pool.join()
    
    return output

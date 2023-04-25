'''
'mandulakenyer' module containts different implementations of Mandelbrot algorithm.
'''

import numpy as np
import multiprocessing as mp

"""
class Mandelbrot():

    def __init__(self, re, im,  I, T):
        self.re = np.linspace(re[0], re[1], re[2], dtype=np.float32)
        self.im = np.linspace(im[0], im[1], im[2], dtype=np.float32)
        self.max_iter = I
        self.thresh = T
    
    def naive():
        pass
"""

def create_imre(re_prop, im_prop):
    '''
    Returns 2 arrays 

    Inputs
        re_prop : list of properties of on real-axis - [minimum, maximum, scale]
        im_prop : list of properties of on imaginary-axis - [minimum, maximum, scale]
    
    Outputs
        re : values on real-axis
        im : values on imaginary-axis
    '''

    re = np.linspace(re_prop[0], re_prop[1], re_prop[2], dtype=np.float32)
    im = np.linspace(im_prop[0], im_prop[1], im_prop[2], dtype=np.float32) *1j
    
    return re, im


def naive(re, im, itr, thresh):
    '''
    Implements a naive version of Mandelbrot algorithm
    Returns a Mandelbrot dataset.

    Inputs
        re : values on real-axis
        im : values on imaginary-axis
        itr : iteration number
        thresh : threshold
    
    Outputs
        M : Mandelbrot dataset
    '''

    M = np.zeros((re.size, im.size))
    
    for a in range(len(re)): 
        for b in range(len(im)):
            z = 0 + 0j
            c = re[a] + im[b]*1j

            for i in range(itr):
                z = z**2 + c
            
                if abs(z) > thresh:
                    M[b, a] = i
                    break
            #else:
                #M[b, a] = itr

    return M

def vectorized(re, im, itr, thresh):
    '''
    Implements a vectorized version of Mandelbrot algorithm
    Returns a Mandelbrot dataset.

    Inputs
        re : list of properties of on real-axis - [minimum, maximum, scale]
        im : list of properties of on imaginary-axis - [minimum, maximum, scale]
        itr : uteration number
        thresh : threshold
    
    Outputs
        M : Mandelbrot dataset
    '''

    # re_val, im_val = create_imre(re, im)
    M = np.zeros((re.size, im.size))
    z = np.zeros((re.size, im.size))
    c = re + im[:, np.newaxis]*1j

    for i in range(itr):
        z = z**2 + c
        M[thresh <= abs(z)] = i

    #M[M == 0] = itr

    return M

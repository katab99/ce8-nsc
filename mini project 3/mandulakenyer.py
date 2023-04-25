'''
'mandulakenyer' module containts different implementations of Mandelbrot algorithm.
'''
import numpy as np

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

    return M

import numpy as np

def parallel_mb(c_matrix, iter, thresh):
    M = np.zeros(c_matrix.shape[0])
    
    for count, value in enumerate(c_matrix):
        c = value
        z = 0 + 0j
        
        for i in range(iter):
            z = z**2 + c
            
            if abs(z) > thresh:
                M[count] = i 
                break
        else:
            M[count] = iter
            
    
    return M

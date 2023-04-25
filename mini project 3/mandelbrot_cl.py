import pyopencl as cl
import numpy as np

def mandelbrot_cl(re, im, max_iter, thresh):
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    
    C = re + im[:,None]*1j
    C = np.ravel(C)
    
    M = np.empty(C.shape, dtype=np.uint16)
    
    kernel = open("kernel.cl").read()

    prg = cl.Program(ctx, kernel).build()

    mf = cl.mem_flags
    C_buff = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=C)
    M_buff = cl.Buffer(ctx, mf.WRITE_ONLY, M.nbytes)

    prg.mandelbrot(queue, M.shape, None, C_buff, M_buff, np.uint16(max_iter), np.float32(thresh))

    cl.enqueue_copy(queue, M, M_buff).wait()
    
    M = M.reshape((re.size, im.size))
    
    return M
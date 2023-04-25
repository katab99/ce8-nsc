__kernel void mandelbrot(
    __global float2 *C,
    __global ushort *M, 
    ushort const max_iter,
    float const thresh)
    {
        int gid = get_global_id(0);
        
        float zr = 0;
        float zi = 0;

        float zr_new, zi_new; 
        
        M[gid] = 0;
        
        for(int i = 0; i < max_iter; i++) {
            zr_new = zr*zr - zi*zi + C[gid].x;
            zi_new = 2*zr*zi + C[gid].y;
            
            zr = zr_new;
            zi = zi_new;

            if (zr*zr + zi*zi > thresh * thresh){
                M[gid] = i;
                break;
            }
        }
    }
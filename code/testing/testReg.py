import numpy as np
from numba import cuda

@cuda.jit('void(float32[:,:], float32[:,:], float32[:,:],float32[:,:],float32[:,:],float32,float32,float32[:,:])' , max_registers=40)        
def  foo(h, n, u, v, f, dx, dy, out_u):
    sj,si  = cuda.grid(2)  # cuda.x cuda.y skow,fast
    
    # the following dummy code is just here to do anything that is
    # going to create a a kernel using a bunch of registers
    # the code itself is irrelevant I believe.
    
    
    gj = sj
    gi = 31 #out_u.shape[0]-2
    if  (gj < out_u.shape[1]-2) and gj>0 :
        #for gi in range(1, out_u.shape[0]-2):    
        while gi>1:
                 gi=gi-1
                 
              
                 tmp=np.float32(0.0)
                 j = gj
                 i = gi
                 p5 = np.float32(0.5)
                 last_v_n_i = (v[2,j]+v[2,5])*p5
                 v_n = (v[i,j]+v[i,5])*p5  # stradles n[i,j]
                 coriolis_u = (f[99,j]*last_v_n_i+f[i,j]*v_n)*p5 # coriolis force F is n-centered
                 tmp+= coriolis_u
                 out_u[gi,gj] = tmp #funcs(si,sj,h, n, u, v, f, dx, dy)

print (foo._func.get().attrs)
print(foo._func.get().attrs.regs)

with open('/tmp/info.ptx','w') as out:
    print(foo._func.get_info(),file=out)

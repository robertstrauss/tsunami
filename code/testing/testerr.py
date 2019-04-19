import numpy as np
import time
import numba as nb
import math
sqrt = math.sqrt
from numba import cuda

device_ii_compiler = cuda.jit('float32(int32,int32,float32[:,:], float32[:,:], float32[:,:],float32[:,:],float32[:,:],float32,float32)',device=True) #maxregisters=4) 

compiler = cuda.jit('void(float32[:,:], float32[:,:], float32[:,:],float32[:,:],float32[:,:],float32,float32,float32[:,:])' ) #,maxregisters=4)        
def compileit (fn,blockdim=None,border=(0,0)):
    name = fn.__name__
    fc = compiler(fn)
    fc.__name__ = name+'cuda'
    fc.BLOCKDIM = blockdim
    fc.BORDER = border
    return fc

def dudt2_dummy_ret_py(i,j,h, n, f, u, v, dx, dy) :
    p5 = np.float32(0.5)
    last_v_n_i = (v[2,j]+v[2,5])*p5
    v_n = (v[i,j]+v[i,5])*p5  # stradles n[i,j]
    coriolis_u = (f[99,j]*last_v_n_i+f[i,j]*v_n)*p5 # coriolis force F is n-centered
    return coriolis_u

cuda_dudt2_dummy_ret=device_ii_compiler(dudt2_dummy_ret_py)

def cu_u_driver_global_funcs_col_py(h, n, u, v, f, dx, dy, out_u):
    sj,si  = cuda.grid(2)  # cuda.x cuda.y skow,fast
    gj = sj
    for gi in range(1, out_u.shape[0]-2):    
        if  (gj < out_u.shape[1]-2) and gj>0 :
            
                 tmp = cuda_dudt2_dummy_ret(gi,gj,h, n, f, u, v, dx, dy) #
                 # now I just add on some more dummy calucaltions to titrate the 
                 # place where the error gets tripped.  
                 
                 tmp += sqrt(abs(np.float32(tmp)*dy+np.float32(gj)*dx))  #  comment or un comment this line
                 tmp += sqrt(abs(np.float32(gi)*dx+np.float32(gj)*dx))  #  comment or un comment this line
#                  tmp += sqrt(abs(np.float32(tmp)*dy+np.float32(gj)*dx))  #  comment or un comment this line
#                  tmp += sqrt(abs(np.float32(gi)*dx+np.float32(gj)*dx))  #  comment or un comment this line
#                  tmp += sqrt(abs(np.float32(tmp)*dy+np.float32(gj)*dx))  #  comment or un comment this line
#                  tmp += sqrt(abs(np.float32(gi)*dx+np.float32(gj)*dx))  #  comment or un comment this line
#                  tmp += sqrt(abs(np.float32(gi)*dy+np.float32(gj)*dx))  #  comment or un comment this line
#                  tmp += sqrt(abs(np.float32(gi)*dx+np.float32(gj)*dx))  #  comment or un comment this line
#                  tmp += sqrt(abs(np.float32(gi)*dy+np.float32(gj)*dx))  #  comment or un comment this line
#                  tmp += sqrt(abs(np.float32(gi)*dx+np.float32(gj)*dx))  #  comment or un comment this line
#               #   tmp += sqrt(abs(gi*dy+gj*dx))  #  comment or un comment this line
        

                 out_u[gi,gj] = tmp #funcs(si,sj,h, n, u, v, f, dx, dy)


cu_u_driver_global_funcs_col = compileit(cu_u_driver_global_funcs_col_py,blockdim=(1024,1))


M=10000
N=1000
h = np.ones((N,M),dtype=np.float32) #np.asarray( np.float32(2)+np.random.random((N,M)) ,dtype=np.float32)
n = np.ones((N,M),dtype=np.float32) #np.asarray(np.float32(2)+np.random.random((N,M)) ,dtype=np.float32)
u =np.ones((N+1,M),dtype=np.float32) # np.asarray(np.random.random((N+1,M)) ,dtype=np.float32)
v = np.ones((N,M+1),dtype=np.float32) #np.asarray(np.random.random((N,M+1)) ,dtype=np.float32)
f = np.ones((N,M),dtype=np.float32) #np.asarray(np.random.random((N,M)) ,dtype=np.float32)
dx = np.float32(0.1)
dy = np.float32(0.2)

out_u = np.zeros_like(u)


def testit(blockspergrid,threadsperblock,cu_u_driver,res1=None ):
            print (cu_u_driver.__name__)
            h1 = cuda.to_device(h)
            n1 = cuda.to_device(n)
            u1 = cuda.to_device(u)
            v1 = cuda.to_device(v)
            f1 = cuda.to_device(f)
            out_u1 =cuda.to_device(out_u)
            
            print( "blocks per grid", blockspergrid)
            print("threads per block",threadsperblock)
            cu_u_driver[blockspergrid,threadsperblock](h1, n1, u1, v1, f1, dx, dy, out_u1)

for i in range(512,1001,4):            
    testit((1,1),(1,i),cu_u_driver_global_funcs_col)

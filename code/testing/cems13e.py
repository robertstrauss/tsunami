import time
from numba import cuda, float32
import numpy as np


print ("wombat")

@cuda.jit('void(float32[:,:], float32[:,:])')
def cudacuda(  v,out_u):
    def cu_u_driver_global( v,out_u):
        """This kernel function will be executed by a thread."""
        for r in range(1):
            i, j  = cuda.grid(2)
            k = np.float32(0.0)
            if (i < out_u.shape[0]-1) and (j < out_u.shape[1]-1):
                if  i>1  and j>1 : 
                    p5 = np.float32(0.5)
                    k = (v[i-1,j]+v[i-1,j+1])*p5
                out_u[i,j]=k    
            cuda.syncthreads()  # maybe not needed
    for i in range(1000):
        cu_u_driver_global( v,out_u)

mytime = time.perf_counter
#mytime = time.time
@cuda.jit('void(float32[:,:], float32[:,:])',device=True)      
def cudadevice(v,out_u):
    i, j  = cuda.grid(2)
    k = np.float32(0.0)
    if (i < out_u.shape[0]-1) and (j < out_u.shape[1]-1):
        if  i>1  and j>1 : 
            p5 = np.float32(0.5)
            k = (v[i-1,j]+v[i-1,j+1])*p5
    out_u[i,j]=k        
    
@cuda.jit('void(float32[:,:], float32[:,:])')      
def cudadevicecaller(v,out_u):
    for yi in range(1000):
        cudadevice(v,out_u)     
            
def testin():
    N=2000  
    M=2000
    h = np.asarray( np.float32(2)+np.random.random((N,M)) ,dtype=np.float32)
    n = np.asarray(np.random.random((N,M)) ,dtype=np.float32)
    u = np.asarray(np.random.random((N+1,M)) ,dtype=np.float32)
    v = np.asarray(np.random.random((N,M+1)) ,dtype=np.float32)
    f = np.asarray(np.random.random((N,M)) ,dtype=np.float32)
    dx = np.float32(0.1)
    dy = np.float32(0.2)
    #p.g = np.float32(1.0)
    nu= np.float32(1.0)
    



    out_u = np.asarray(np.random.random((M,N+1)) ,dtype=np.float32)


    threadsperblock = (16,32) # (16,16)
    blockspergrid_x = (u.shape[0] + threadsperblock[0]) // threadsperblock[0]
    blockspergrid_y = (u.shape[1] + threadsperblock[1]) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    print ("here we go",u.shape)
    print( "blocks per grid", blockspergrid)
    print("threads per block",threadsperblock)
    v1 = cuda.to_device(v)
#           f1 = cuda.to_device(f)
    out_u1 =cuda.to_device(out_u)
    ts = []
    for i in range(10):
        t = mytime()  # time.process_time()
        cudadevicecaller[blockspergrid,threadsperblock](v1,  out_u1)
        cuda.synchronize()    
        t2 =  mytime()  # time.process_time()
        ts.append(t-t2)
      #  time.sleep(1)
    print("cudacall") 
    print (np.median(ts),np.min(ts),np.max(ts),np.std(ts))
    print (ts)
    print ("all done")
#     for cu_u_driver in (cu_u_driver_global,):
#         print (cu_u_driver)
# #           h1 = cuda.to_device(h)
# #           n1 = cuda.to_device(n)
# #           u1 = cuda.to_device(u)
#         v1 = cuda.to_device(v)
# #           f1 = cuda.to_device(f)
#         out_u1 =cuda.to_device(out_u)
#         ts = []
#         for i in range(10):
#             t = mytime()  # time.process_time()
#             for j in range(100):
#                 cu_u_driver[blockspergrid,threadsperblock](v1,  out_u1)
#                 cu_u_driver[blockspergrid,threadsperblock](v1,  out_u1)
#                 cu_u_driver[blockspergrid,threadsperblock](v1,  out_u1)
#                 cu_u_driver[blockspergrid,threadsperblock](v1,  out_u1)
#                 cu_u_driver[blockspergrid,threadsperblock](v1,  out_u1)
#                 cu_u_driver[blockspergrid,threadsperblock](v1,  out_u1)
#                 cu_u_driver[blockspergrid,threadsperblock](v1,  out_u1)
#                 cu_u_driver[blockspergrid,threadsperblock](v1,  out_u1)
#                 cu_u_driver[blockspergrid,threadsperblock](v1,  out_u1)
#                 cu_u_driver[blockspergrid,threadsperblock](v1,  out_u1)
#                 
#             cuda.synchronize()    
#             t2 =  mytime()  # time.process_time()
#             ts.append(t-t2)
#           #  time.sleep(1)
#         print("cuda") 
#         print (np.median(ts),np.min(ts),np.max(ts),np.std(ts))
#         print (ts)
#     
#    

testin()

import numba as nb
import numpy as np
from numba import cuda

@cuda.jit('void(float32[:], float32[:], float32[:])')
def cu_add1(a, b, c):
    """This kernel function will be executed by a thread."""
    bx = cuda.blockIdx.x # which block in the grid?
    bw = cuda.blockDim.x # what is the size of a block?
    tx = cuda.threadIdx.x # unique thread ID within a blcok
    i = tx + bx * bw

    if i > c.size:
        return

    c[i] = a[i] + b[i]
    
    
device = cuda.get_current_device()

n = 100

# Host memory
a = np.arange(n, dtype=np.float32)
b = np.arange(n, dtype=np.float32)

# Assign equivalent storage on device
da = cuda.to_device(a)
db = cuda.to_device(b)

# Assign storage on device for output
dc = cuda.device_array_like(a)

# Set up enough threads for kernel
tpb = device.WARP_SIZE
bpg = int(np.ceil(float(n)/tpb))
print ('Blocks per grid:', bpg)
print ('Threads per block', tpb)

# Launch kernel
cu_add1[bpg, tpb](da, db, dc)

# Transfer output from device to host
c = dc.copy_to_host()

print (c)    
#
@cuda.jit('void(float32[:,:], float32[:,:], float32[:,:])')
def cu_add_2d(a, b, c):
    """This kernel function will be executed by a thread."""
    i, j  = cuda.grid(2)

    if (i < c.shape[0]) and (j < c.shape[1]):
        c[i, j] = a[i, j] + b[i, j]
    cuda.syncthreads()
    
    
device = cuda.get_current_device()

n = 480
p = 320
a = np.random.random((n, p)).astype(np.float32)
b = np.ones((n, p)).astype(np.float32)
c = np.empty_like(a)

threadsperblock = (16, 16)
blockspergrid_x = (n + threadsperblock[0]) // threadsperblock[0]
blockspergrid_y = (p + threadsperblock[1]) // threadsperblock[1]
blockspergrid = (blockspergrid_x, blockspergrid_y)

print (blockspergrid, threadsperblock)

cu_add_2d[blockspergrid, threadsperblock](a, b, c)
print (a[-5:, -5:])
print (b[-5:, -5:])
print (c[-5:, -5:])


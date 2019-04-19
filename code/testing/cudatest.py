
# coding: utf-8

# In[1]:


import numpy as np
import numba as nb
from numba import cuda, float32


# In[16]:
N=M=16

# Controls threads per block and shared memory usage.
# The computation will be done on blocks of TPBxTPB elements.
bw = 16 #cuda.blockDim.x
bh = 16 # cuda.blockDim.y


#@cuda.jit(device=False)

bpg = 50
tpb = 32
n = bpg * tpb

@cuda.jit(argtypes=[float32[:,:], float32[:,:], float32[:,:]], target='gpu')
def fast_matmul(A, B, C):
    sA = cuda.shared.array(shape=(tpb, tpb), dtype=float32)
    sB = cuda.shared.array(shape=(tpb, tpb), dtype=float32)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    bw = cuda.blockDim.x
    bh = cuda.blockDim.y

    x = tx + bx * bw
    y = ty + by * bh

    acc = 0.
    for i in range(bpg):
        if x < n and y < n:
            sA[ty, tx] = A[y, tx + i * tpb]
            sB[ty, tx] = B[ty + i * tpb, x]

        cuda.syncthreads()

        if x < n and y < n:
            for j in range(tpb):
                acc += sA[ty, j] * sB[j, tx]

        cuda.syncthreads()

    if x < n and y < n:
        C[y, x] = acc

@cuda.jit
def matmul(A, B, C):
    """Perform square matrix multiplication of C = A * B
    """
    i, j = cuda.grid(2)
    if i < C.shape[0] and j < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]
        C[i, j] = tmp


# In[12]:



A = np.zeros((5,5),dtype=np.float32)  
i = np.arange(A.shape[0])
A[i,i]=2
B=np.zeros_like(A)
C=np.ones_like(B)
C[0,0] = 8
C[-1,-1]=9
B[i,(i+1)%A.shape[1]] = 3
print(A)
print(B)



threads_per_block=(bw,)
xblocks = np.int( (A.shape[0]+threads_per_block[0]-1)/threads_per_block[0])
yblocks = np.int( (B.shape[1]+threads_per_block[0]-1)/threads_per_block[0]) 

blocks_per_grid=(xblocks,)


A=nb.SmartArray(A)
B=nb.SmartArray(B)
C=nb.SmartArray(C)


#A_image = cuda.to_device(A)
#B_image = cuda.to_device(B)
#C_image = cuda.to_device(C)

# In[13]:

print(blocks_per_grid)
print(threads_per_block)

# In[17]:

fast_matmul[(bpg,),(tpb,)](A,B,C)
#C_image.to_host()
print(np.asarray(C))
#print(np.matmul[blocks_per_grid,threads_per_block](B,B.T))
matmul[(bpg,),(tpb,)](A,B,C)
print(np.asarray(C))



# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import numpy as np
import cupy as qp
import operator as op
import time
import matplotlib as mpl
import pandas as pd
from pandas import HDFStore, DataFrame
from matplotlib import animation, rc
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interactive, Button
from IPython.display import display, HTML
import netCDF4 as nc
import numba as nb

xp = np
argfast = {'parallel':True, 'fastmath':True, 'nopython':True}
argfast_noparallel = {'parallel':False, 'fastmath':True, 'nopython':True}


# In[2]:



#convenient floats
zero = np.float32(0)
p5 = np.float32(0.5)
one = np.float32(1)
two = np.float32(2)


# physics constants
class p():
    g = np.float32(9.81) # gravity
    mu = np.float32(0.3)
  
zero = np.float32(0)
p5 = np.float32(0.5)
one = np.float32(1)
two = np.float32(2)


# physics constants
#g = np.float32(9.81) # gravity
#mu = np.float32(0.3)
  


# In[3]:


# display functions

def disp3d(fig, aa, lines=(35,35)): # 3d wirefram plot
    # interpert inputs
    aa = qp.asnumpy(aa)
    
#     xlim = box[0]
#     ylim = box[1]
#     zlim = box[2]
#     if (xlim==None):
#         xlim = (0, aa[0].shape[0])
#     if (ylim==None): ylim = (0, aa[0].shape[1])
#     if (zlim==None):
#         ran = np.max(aa[0])-np.min(aa[0])
#         zlim = (np.min(aa[0])-ran, np.max(aa[0])+ran)
#         zlim = (-2, 2)
    
    #'wires' of the wireframe plot
    x = np.linspace(0, aa[0].shape[0]-1, lines[0], dtype=int)
    y = np.linspace(0, aa[0].shape[1]-1, lines[1], dtype=int)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    
    #display it
#     fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
#     for a in aa:
    A = aa#[xx,yy]
    ax.plot_wireframe(xx, yy, A)
    
    return ax



# def genframes(a, frames=120):#np.arange(120, dtype=int)):
#     arts = np.array([])#np.empty((frames.shape[0],))
# #     assert frames.dtype==int, "frames must be a numpy array of integers"
# #     frames = np.asarray(frames, dtype=int)
#     mm = np.max([-np.min(a), np.max(a)])/2
#     ds = a[np.linspace(0, a.shape[0]-1, frames, dtype=int)]
#     for d in ds:
# #         d = np.asarray(a[frame], dtype=np.float32) # data
#         im = plt.imshow(d, animated=True, vmin=-mm, vmax=mm, cmap='seismic')
#         arts = np.append(arts, im)
#     return f

# def motioncon(fig, f): # animated height plot, takes in list of 2d height arrays
    #prepare figure/display
    
#     z = qp.asnumpy(f[0])
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     mm = np.max([-np.min(f), np.max(f)])/2
#     im = ax.imshow(z, vmin=-mm, vmax=mm,cmap='seismic')
#     cb = fig.colorbar(im)
#     tx = ax.set_title(title)
#     plt.xticks(np.linspace(0, z.shape[0], xlabels.shape[0]), xlabels)
#     plt.yticks(np.linspace(0, z.shape[1], ylabels.shape[0]), ylabels)
    
#     def animate(i): # returns i'th element (height array) in f
#         im.set_data(qp.asnumpy(f[i]))
#         plt.contour(h, levels=1, cmap='gray')
    
    #display it
#     anim = animation.ArtistAnimation(fig, f)
#     return anim

def vect(u, v, arws=(10, 10), arwsz=100): # vector /motion plot
    #interpert inputs
    u = qp.asnumpy(u)
    v = qp.asnumpy(v)
#     if (xlim=='default'): xlim = (0, u.shape[0])
#     if (ylim=='default'): ylim = (0, v.shape[1])
    arws = (int(arws[0]), int(arws[1]))
    
    # set up
    x = np.linspace(0, u.shape[0]-1, arws[0], dtype=int)
    y = np.linspace(0, v.shape[1]-1, arws[1], dtype=int)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    uu = u[x,y]
    vv = v[x,y]
    m = np.hypot(uu, vv)
    
    #displat it
    q = plt.quiver(xx, yy, uu, vv, m, scale = 1/arwsz)
#     return ax
#     ax.set_xlim(xlim)
#     ax.set_ylim(ylim)
#     plt.title(title)
#     plt.show()


# In[4]:


# useful math functions
@nb.jit(**argfast) # (parallel=True)
def d_dx(a, dx):
    ddx = ( a[:-1] - a[1:] )*(np.float32(-1)/dx) 
    return ddx
@nb.jit(**argfast) # (parallel=True)
def d_dy(a, dy):
    ddy = ( a[:,:-1] - a[:,1:] )*(np.float32(-1)/dy)
    return ddy
@nb.jit(**argfast) # (parallel=True)
def div(u, v, dx, dy):  
    div = d_dx(u, dx) + d_dy(v, dy)
    return div

# return result in usder supplied matrix
@nb.jit(**argfast) # (parallel=True)
def d_dx0(a, dx, out):  
    '''d_dx(a, dx,out) for a[n,m] then out[n-1,m]'''
    out[:,:] = ( a[:-1] - a[1:] )*(np.float32(-1)/dx) 
    return out
@nb.jit(**argfast) # (parallel=True)    
def d_dy0(a, dy,out):
    '''d_dy(a, dx,out) for a[n,m] then out[n,m-1]'''
    out[:,:] = ( a[:,:-1] - a[:,1:] )*(np.float32(-1)/dy)
    return out
@nb.jit(**argfast) # (parallel=True)    
def div0(u, v, dx, dy,out):
    '''div(u,v,dx,dy,out,tmp): u(n+1,m)  v(n,m+1) out[n,m] out is retun,'''
    out[:,:] = d_dx0(u,dx) + d_dy0(v, dy)

    
def d_dx1(a, dx, out):  
    '''d_dx(a, dx,out) for a[n,m] then out[n-1,m]'''
    for i in nb.prange(a.shape[0]-1):
        for j in nb.prange(a.shape[1]):
            out[i,j]= (a[i+1,j]-a[i,j])/dx[i,j]
    return out

def d_dy1(a, dy, out):  
    '''d_dx(a, dx,out) for a[n,m] then out[n-1,m]'''
    for i in nb.prange(a.shape[0]):
        for j in nb.prange(a.shape[1]-1):
            out[i,j]= (a[i,j+1]-a[i,j])/dy[i,j]
    return out
@nb.jit(**argfast) # (parallel=True)  
def d_dx_d_dy(u,v, dx,dy, outx,outy):  
    '''computes both du_dx and  du_dy in the same loop
       input for u[n+1,m] and v[n,m+1]
       then outx[n,m] outy[n,m] '''
    for i in nb.prange(v.shape[0]):
        for j in nb.prange(u.shape[1]):
            outx[i,j]= (u[i+1,j]-u[i,j])/dx[i,j]
            outy[i,j]= (v[i,j+1]-v[i,j])/dy[i,j]
    return outx,outy
    
    
@nb.jit(**argfast) # (parallel=True)    
def div1(u,v,dx,dy,out):
    ''' explicit loops and assumes dx and dy are matricies'''
    for i in nb.prange(v.shape[0]):
        for j in nb.prange(u.shape[1]):
            # note U and V are shaped  u[n+1,m]  v[n,m+1]
            out[i,j]= ( u[i+1,j] -u[i,j]  )/dx[i,j] + ( v[i,j+1] -v[i,j]  )/dy[i,j]
    return out  # this is superfluous as out is modified inplace       


# In[5]:


@nb.jit(**argfast) # (parallel=True) 
def sum_x(in2d,out2d):
    ''' sums x-neighbors in2d[n,m] out2d[n-1,m]'''
    out2d[:,:] = in2d[1:,:] + in2d[:-1,:]


@nb.jit(**argfast) # (parallel=True)     
def diff_x(in2d,out2d):
    ''' diferences x-neighbors in2d[n,m] out2d[n-1,m]'''
    out2d[:,:] = in2d[1:,:] - in2d[:-1,:]


@nb.jit(**argfast) # (parallel=True)     
def sum_y(in2d,out2d):
    ''' sums y-neighbors in2d[n,m] out2d[n,m-1]'''
    out2d[:,:] = in2d[:,1:] +in2d[:,:-1]


@nb.jit(**argfast) # (parallel=True) 
def diff_y(in2d,out2d):
    ''' diferences y-neighbors in2d[n,m] out2d[n,m-1]'''
    out2d[:,:] = in2d[:,1:]-in2d[:,:-1]

 


# In[6]:


# for generating simple environments or initial conditions
# def hydrodynamic
def planegauss(shape, win=((-2, 2), (-2, 2))):
   # h=np.empty(shape, dtype=np.float32)
    npx = np.linspace( win[0][0], win[0][1], shape[0] )
    npy = np.linspace( win[1][0],win[1][1], shape[1] )
    npxx, npyy = np.meshgrid(npx, npy, indexing='ij')
    h = np.exp( -np.e * ( npxx*npxx + npyy*npyy ) )
    return (h)
def lingauss(shape, w = 1/2, ax = 0, win = (-2, 2)):
   # h=np.empty(shape, dtype=np.float32)
    npx = np.linspace( win[0], win[1], shape[0] )
    npy = np.linspace( win[0], win[1], shape[1] )
    npxx, npyy = np.meshgrid(npy, npx)
    xy = (npyy, npxx)[ax]
    h = np.exp( -np.e * ( xy*xy ) / (w*w) )
    return (h)


# In[7]:



class State(): # state
    def __init__(self, h, n, u, v, dx, dy, lat, lon):
        
        self.dx = dx  # this is v-centered
        self.dy = dy  # this is u-centered
        self.lat = lat
        self.lon = lon
#         self.lats, self.lons = np.meshgrid(self.lat, self.lon)
#         self.lat, self.lon = np.meshgrid(self.lat, self.lon) # lattitude/longitude chunk simulation area stretches over
        self.h = h
    
        self.maxws = np.sqrt(np.max(self.h)*p.g) # maximum wave speed
        
        self.n = np.asarray(n, dtype=np.float32) # surface height (eta)
        self.u = np.asarray(u, dtype=np.float32) # x vel array
        self.v = np.asarray(v, dtype=np.float32) # y vel array
        
        #make sure h is the same shap as n (eta)
        assert (np.isscalar(h) or self.h.shape == self.n.shape), "H and N must have the same shape, or H must be a scalar" # 'or' is short circuit
        
#         self.calcDt()
        self.dt = np.min((np.min(self.dx), np.min(self.dy)))/(5*self.maxws)
        
        self.coriolis = np.float32(((2*2*np.pi*np.sin(self.lat*np.pi/180))/(24*3600))[:,np.newaxis]) # rotation speed of the earth dtheta/dt
        """ derivation of coriolis force
        U = R*cos(phi)*O
        ui = U+ur
        ur = ui-U
        dU/dphi = -R*sin(phi)*O
        phi = y/R
        dphi/dt = v/R
        dU/dt = v*(-sin(phi)*O)
        dur/dt = dui/dt - dU/dt = v*O*sin(phi)
        dur/dt = v*O*sin(phi)"""
        self.movetodevice()
   
    def movetodevice(self):
        self.lat = xp.asarray(self.lat,dtype=np.float32)
        self.lon = xp.asarray(self.lon,dtype=np.float32)
        self.h = xp.asarray(self.h,dtype=np.float32)
        self.n = xp.asarray(self.n,dtype=np.float32)
        self.u = xp.asarray(self.u,dtype=np.float32)
        self.v = xp.asarray(self.v,dtype=np.float32)
        self.coriolis = xp.asarray(self.coriolis,dtype=np.float32)
#     def calcDt(self, fudge = 5): #calculate optimal value of dt for the height and dx values
#         dx = np.min(self.dx)
#         dy = np.min(self.dy)
#         self.dt = np.min((dx, dy))/(fudge*self.maxws)
props = op.itemgetter('h', 'n', 'u', 'v', 'dx', 'dy', 'lat', 'lon') # for grrabbing the elements of a state
# def newstate(state):
#     return State(*(props(vars(state))))


# # physics shallow water framework

# ## Class of objects to hold current state of an ocean grid 
# 
# Equations of motion
# $$
# \begin{align}
# \frac{\partial \eta}{\partial t} & =
#     -\frac{\partial  }{\partial x} \bigl( \left( \eta + h\right)u \bigr) 
#     - \frac{\partial  }{\partial y}  \bigl( \left( \eta + h\right)v \bigr)\\  
# \\
# \frac{\partial u}{\partial t} & = Coriolis + Advection + Gravity + Attenuation\\
#  & = +fv +\bigl( \kappa\nabla^{2}u - (u,v)\cdot\vec\nabla u \bigr)  
#     - g\frac{\partial \eta}{\partial x} - \frac{1}{\rho (h + \eta)} \mu u \sqrt{u^{2} + v^{2}}\\  
# & = +fv +\bigl( \kappa\frac{\partial^{2} u}{\partial x^{2}}
#            +\kappa\frac{\partial^{2} u}{\partial y^{2}}
#            -u\frac{\partial u}{\partial x} - v\frac{\partial u}{\partial y}\bigr) 
#            - g\frac{\partial \eta}{\partial x}
#             - \frac{1}{\rho (h + \eta)} \mu u \sqrt{u^{2} + v^{2}}\\
# \\
# \frac{\partial v}{\partial t} & = -fu 
#    + \bigl( \kappa\nabla^{2}v - (u,v)\cdot\vec\nabla v \bigr) 
#     - g\frac{\partial \eta}{\partial y}
#     - \frac{1}{\rho (h + \eta)} \mu v \sqrt{u^{2} + v^{2}}\\   
# & = -fu+\bigl( \kappa\frac{\partial^{2} v}{\partial x^{2}}
#            +\kappa\frac{\partial^{2} v}{\partial y^{2}}
#            -u\frac{\partial v}{\partial x} - v\frac{\partial v}{\partial y}\bigr) 
#            - g\frac{\partial \eta}{\partial y}
#            - \frac{1}{\rho (h + \eta)} \mu v \sqrt{u^{2} + v^{2}}\\           
# \end{align}
# $$
# 
# Where 
# - *_h_* calm ocean depth (positive number) at any point. Presumed constant in time
# - $\eta$ is the wave height above the calm ocean height
# - *_u_* is the mean water column velocity in the _x_ (east) direction
# - *v* is the mean water column velocity in the _y_ (north) direction
# 
# and the physcial constant parameters are:
# - *g* gravitational constant
# - *f* is the lattidude dependent coriolis coefficient: $2\omega \sin(latitude)$
# - $\kappa$ is the viscous damping coefficient across the grid cell boundaries
# - $\mu$ is the friction coeffecient
# 

# In[8]:


@nb.jit(**argfast)  # never use parallel if there is a broadcast-- numba bug 
def dndt(h, n, u, v, dx, dy) : 
    """change in n per timestep, by diff. equations"""
    p5=np.float32(0.5) # needs to be her for numba
    hx = xp.empty(u.shape, dtype=n.dtype) # to be x (u) momentum array
    hy = xp.empty(v.shape, dtype=n.dtype)
    
    depth = h+n
    hx[1:-1] = (depth[1:] + depth[:-1])*p5 # average
    hx[0] = zero # normal flow boundaries/borders       #  is this actually wrong ####################
    hx[-1] = zero 
    
    hy[:,1:-1] = (depth[:,1:] + depth[:,:-1])*p5
    hy[:,0] = zero
    hy[:,-1] = zero
    #print ("depth",hx[2,2],hx[3,2],hy[2,2],hy[2,3])
    hx *= u # height/mass->momentum of water column.
    hy *= v
   # print("depth*u",hx[2,2],hx[3,2],hy[2,2],hy[2,3])
   # print ("depth*u/dx",(hx[2,2]-hx[3,2])/dx,(hy[2,2]-hy[2,3])/dy)
#     dndt = d_dx(hx, -dx)+d_dy(hy, -dy)#(div(hx, hy, -dx, -dy))
    return ( d_dx(hx, -dx) +d_dy(hy, -dy ) )

@nb.njit()   # never use parallel if there is a broadcast-- numba bug  
def dndt2(i,j,h, n, u, v, dx, dy) : # for individual vars
    """change in n per timestep, by diff. equations
    u[n+1,m] v[n,m+1] n[n,m] h[n,m]
    out[n,m] 
    use edge functions for i=0(-1) j=0(-1)
     as written this malfunctions when:
         i=0 or j=0:  causes wrap around to i=-1 
         i=n-1 or j=m-1:  tries to retrieve n[n]  and gets out of bounds error
    """
    dn_dt = dndt2_x(i,j,h, n, u, dx )+dndt2_y(i,j,h, n, v, dy )
   
    return dn_dt # minus or plus??
# these are utility functions not called directly by user
@nb.njit()   # never use parallel if there is a broadcast-- numba bug 
def dndt2_x(i,j,h, n, u, dx ) : # for individual vars)
        # as written this malfunctions when:
    # i=0 :  causes wrap around to i=-1
    # i=n-1 :  tries to retrieve n[n]  and gets out of bounds error
    p5=np.float32(0.5)
    last_depth_x = n[i-1,j]+h[i-1,j]  # this economy is bad!  screws up parallizing. remove it
    depth = n[i,j]+h[i,j]
    next_depth_x = n[i+1,j]+h[i+1,j]
    dn_dt_x= (u[i+1,j]*(next_depth_x + depth)*p5 - u[i,j]*(last_depth_x + depth)*p5)/dx
    return dn_dt_x

@nb.njit()   # never use parallel if there is a broadcast-- numba bug 
def dndt2_y(i,j,h, n, v, dy ):
    # equivalent to dndt2_y_mirror
    return dndt2_x(i=j,j=i,h=h.T, n=n.T, u=v.T, dx=dy )


@nb.njit()  # never use parallel if there is a broadcast-- numba bug  
def dndt2_niedge(j,h, n, u, v, dx, dy):
    """ Use when i=-1"""
    
    i=n.shape[0]-1
    dn_dt = dndt2_x_nedge(j,h, n, u, dx=dx ) 
   
    if j==0:
        dn_dt += dndt2_y_0edge(i,h, n, v, dy=dy ) #special speical case
    elif j==n.shape[1]-1:
        dn_dt  += dndt2_y_medge(i,h, n, v, dy=dy )  #dndt2_mjedge(i,h, n, v, dy=dy )
    else:
        dn_dt += dndt2_y(i,j,h, n, v,  dy=dy ) 
    return dn_dt

@nb.njit()   # never use parallel if there is a broadcast-- numba bug  
def dndt2_0iedge(j,h, n, u, v, dx, dy):
    """use when i=0 """
    
    i=0
    dn_dt = dndt2_x_0edge(j,h, n, u, dx=dx )
   
    if j==0 :
        dn_dt += dndt2_y_0edge(i,h, n, v, dy=dy ) 
        
    elif j==n.shape[1]-1 :
        dn_dt  += dndt2_y_medge(i,h, n, v, dy=dy )
    else:
        dn_dt += dndt2_y(i,j,h, n, v,  dy=dy ) 
       # print(j,dn_dt,"0iedge", dndt2_x_0edge(j,h, n, u, dx=dx ),dndt2_y(i,j,h, n, v,  dy=dy )  )
    return dn_dt

# don't need the mirror functions because I can just use transposes! note dx and dy , us uy flipped order. i becomes j

@nb.njit()   # never use parallel if there is a broadcast-- numba bug  
def  dndt2_0jedge(i,h, n, u, v, dx,dy):
    """use when j=0"""
   #equivalent to dndt2_0jedge_mirror(i,h, n, u, v, dx,dy)
    return dndt2_0iedge(j=i,h=h.T, n=n.T,  u=v.T, v= u.T, dx=dy,dy=dx)

@nb.njit()    # never use parallel if there is a broadcast-- numba bug  
def  dndt2_mjedge(i,h, n, u, v, dx,dy):
    """use when j=-1"""
    #equivalent to dndt2_mjedge_mirror(i,h, n, u, v, dx, dy)
    return dndt2_niedge(j=i,h=h.T, n=n.T,  u=v.T, v= u.T, dx=dy,dy=dx)

@nb.njit()    # never use parallel if there is a broadcast-- numba bug  
def dndt2_0jedge_mirror(i,h, n, u, v, dx,dy):
    """use when j=0"""
    j=0
    dn_dt = dndt2_y_0edge(i,h, n, v, dy=dy )
    
    if i==0 :
        dn_dt += dndt2_x_0edge(j,h, n, u, dx=dx ) 
    elif i==n.shape[0]-1 :
        dn_dt  += dndt2_x_nedge(j,h, n, u, dx=dx )
    else:
        dn_dt += dndt2_x(i,j,h, n, u, dx=dx ) 
        #print(i,dn_dt,"0jedge", dndt2_y_0edge(i,h, n, v, dy=dy ), dndt2_x(i,j,h, n, u, dx=dx ) )    
    return dn_dt

@nb.njit()    # never use parallel if there is a broadcast-- numba bug  
def dndt2_mjedge_mirror(i,h, n, u, v, dx, dy):
   
    j=n.shape[1]-1
    dn_dt = dndt2_y_medge(i,h, n, v, dy=dy )
    
    if i==0 :
        dn_dt += dndt2_x_0edge(j,h, n, u, dx=dx ) 
    elif i==n.shape[0]-1 :
        dn_dt  += dndt2_x_nedge(j,h, n, u, dx=dx )
    else:
        dn_dt += dndt2_x(i,j,h, n, u,  dx=dx ) 
    return dn_dt




@nb.njit()    # never use parallel if there is a broadcast-- numba bug  
def dndt2_y_mirror(i,j,h, n, v, dy ) : # for individual vars)
        # as written this malfunctions when:
    # j=0:  causes wrap around to i=-1
    # j=m-1:  tries to retrieve n[n]  and gets out of bounds error
    p5=np.float32(0.5)
    last_depth_y = n[i,j-1]+h[i,j-1]  
    depth = n[i,j]+h[i,j]  # redundant given previous call.
    next_depth_y = n[i,j+1]+h[i,j+1]
    dn_dt_y  =  (v[i,j+1]*(next_depth_y + depth)*p5 - v[i,j]*(last_depth_y + depth)*p5)/dy 
    return dn_dt_y


@nb.njit()    # never use parallel if there is a broadcast-- numba bug  
def dndt2_x_nedge(j,h, n, u, dx ) : # for individual vars)
    p5=np.float32(0.5)
    i=n.shape[0]-1
    # since u[n,j] = 0 (assumed reflective boundary) then we don't care about next_dpeth
    last_depth_x = n[i-1,j]+h[i-1,j]  # this economy is bad!  screws up parallizing. remove it
    depth = n[i,j]+h[i,j]
    #next_depth_x = h[i,j]  # kludge to approximate n[i+1,0 ]=0
    dn_dt_x= ( - u[i,j]*(last_depth_x + depth)*p5)/dx
    return dn_dt_x

@nb.njit()    # never use parallel if there is a broadcast-- numba bug  
def dndt2_x_0edge(j,h, n, u, dx ) : # for individual vars)
    p5=np.float32(0.5)
    i=0 
        # since u[0,j] = 0 (assumed reflective boundary) then we don't care about last_dpeth
    depth = n[i,j]+h[i,j]
    next_depth_x = n[i+1,j]+h[i+1,j]
    dn_dt_x= (u[i+1,j]*(next_depth_x + depth)*p5 )/dx
    return dn_dt_x

@nb.njit()    # never use parallel if there is a broadcast-- numba bug  
def dndt2_y_0edge(i,h, n, v, dy ) : # for individual vars)
    p5=np.float32(0.5)
    j=0
        # since v[i,0] = 0 (assumed reflective boundary) then we don't care about next_dpeth
   # last_depth_y = h[i,j] # edge kludge  
    depth = n[i,j]+h[i,j]  # redundant given previous call.
    next_depth_y = n[i,j+1]+h[i,j+1]     
    dn_dt_y  =  (v[i,j+1]*(next_depth_y + depth)*p5 )/dy
    return dn_dt_y
@nb.njit()    # never use parallel if there is a broadcast-- numba bug  
def dndt2_y_medge(i,h, n, v, dy ) : # for individual vars)
    j=n.shape[1]-1
        # since v[i,m] = 0 (assumed reflective boundary) then we don't care about next_dpeth
    last_depth_y = n[i,j-1]+h[i,j-1]  
    depth = n[i,j]+h[i,j]  # redundant given previous call.
  #  next_depth_y = h[i,j]  # edge kludge
        
    dn_dt_y  =  ( - v[i,j]*(last_depth_y + depth)*p5)/dy
    return dn_dt_y

# In[9]:


@nb.jit(**argfast_noparallel)
def dudt(h, n, f, u, v, dx, dy, nu, mu=p.mu,g=p.g ) : 
    p5=np.float32(0.5)
    dudt = xp.empty(u.shape, dtype=u.dtype) # x accel array
    grav = d_dx(n, -dx/g)
    dudt[1:-1] = grav
#     dudt[0] = grav[0]*2-grav[1] # assume outside map wave continues with constant slope
#     dudt[-1] = grav[-1]*2-grav[-2]
    
    
    # coriolis force
    vn = (v[:,1:]+v[:,:-1])*p5 # n shaped v
    
    fn = f#(f[:,1:]+f[:,:-1])*0.5 # n shaped f

    fvn = (fn*vn) # product of coriolis and y vel.
    dudt[1:-1] += (fvn[1:]+fvn[:-1])*p5 # coriolis force
    dudt[0] += fvn[0]
    dudt[-1] += fvn[-1]
    
    
    # advection
    
    # advection in x direction
    dudx = d_dx(u, dx)
    
    dudt[1:-1] -= u[1:-1]*(dudx[1:] + dudx[:-1])*p5 # advection
#     dudt[0] -= u[0]*dudx[0]
#     dudt[-1] -= u[-1]*dudx[-1]
    
    # advection in y direction
    duy = xp.empty(u.shape, dtype=u.dtype)
    dudy = d_dy(u, dy)
    duy[:,1:-1] = ( dudy[:,1:] + dudy[:,:-1] ) * p5
    duy[:,0] = dudy[:,0]
    duy[:,-1] = dudy[:, -1]
    dudt[1:-1] -= (vn[1:]+vn[:-1])*p5*duy[1:-1] # advection
#     dudt[0] -= vn[0]*duy[0]
#     dudt[-1] -= vn[-1]*duy[-1] # closest to applicable position
    
    mu = np.float32(mu)
    #attenuation
    una = (u[1:]+u[:-1]) * p5
    vna = (v[:,1:]+v[:,:-1])*p5
    attenu = 1/(h+n) * mu * una * np.sqrt(una*una + vna*vna) # attenuation
    dudt[1:-1] -= (attenu[1:] + attenu[:-1])*p5
    
    # viscous term
#     nu = np.float32(1000/dx)

#     ddux = d_dx(dudx, dx)
#     dduy = xp.empty(u.shape, dtype=u.dtype)
#     ddudy = d_dy(duy, dy)
#     dduy[:,1:-1] = ( ddudy[:,1:] + ddudy[:,:-1] ) * p5
#     dduy[:,0] = ddudy[:,0]
#     dduy[:,-1] = ddudy[:, -1]
#     dudt[1:-1] -= nu*(ddux+dduy[1:-1])
    
#     dudt[0] += nu*ddux[0]*dduy[0]
#     dudt[-1] += nu*ddux[-1]*dduy[-1]
    
    dudt[0] = zero
    dudt[-1] = zero # reflective boundaries
    dudt[:,0] = zero
    dudt[:,-1] = zero # reflective boundaries
    return ( dudt )  


# In[10]:


@nb.jit(**argfast_noparallel)
def dvdt(h, n, f, u, v, dx, dy, nu, mu=p.mu,g=p.g) :

    p5=np.float32(0.5)
  
    dvdt = xp.empty(v.shape, dtype=v.dtype) # x accel array

    
    # coriolis force

    un = (u[1:,:]+u[0:-1,:])*p5 # n-shaped u
    unf= un*f# product of coriolis and x vel.
    dvdt[:,1:-1] = (unf[:,1:]+unf[:,:-1])*(-p5) # coriolis force
 



    mu = np.float32(mu)


    grav = d_dy(n, -dy/g)

    dvdt[:,1:-1] += grav


    # advection
    
    # advection in y direction
    dvdy = d_dy(v, dy)

    dvdt[:,1:-1] -= v[:,1:-1]*(dvdy[:,1:] + dvdy[:,:-1])*p5 # advection
#     dvdt[:,0] -= v[:,0]*dvdy[:,0]
#     dvdt[:,-1] -= v[:,-1]*dvdy[:,-1]
    
    # advection in x direction
    dvx = xp.empty(v.shape, dtype=v.dtype)
    dvdx = d_dx(v, dx)
 
    dvx[1:-1] = ( dvdx[1:] + dvdx[:-1] ) * p5
    dvx[0,:] = dvdx[0,:]
    dvx[-1,:] = dvdx[-1,:]
    dvdt[:,1:-1] -= (un[:,1:]+un[:,:-1])*p5*dvx[:,1:-1] # advection
#     dvdt[:,0] -= un[:,0]*dvx[:,0]
#     dvdt[:,-1] -= un[:,-1]*dvx[:,-1] # closest to applicable position
    
    
    una = (u[1:,:]+u[:-1,:]) * p5
    vna = (v[:,1:]+v[:,:-1])*p5
    if np.any( np.isnan(una)): print ("una nan",una)
    if np.any( np.isnan(vna)): print ("vna nan",vna)
    if np.any( np.isinf(una)): print ("una inf",una)
    if np.any( np.isinf(vna)): print ("vna inf",vna)
    attenu = 1/(h+n) * mu * vna * np.sqrt(una*una + vna*vna) # attenuation
    dvdt[:,1:-1] -= (attenu[:,1:] + attenu[:,:-1])*p5
    
    
    # viscous term
#     nu = np.float32(dy/1000) # nu given as argument

#     ddvy = d_dy(dvdy, dy)
#     ddvx = xp.empty(v.shape, dtype=v.dtype)
#     ddvdx = d_dx(dvx, dx)
#     ddvx[1:-1] = ( ddvdx[1:] + ddvdx[:-1] ) * p5
#     ddvx[0] = ddvdx[0]
#     ddvx[-1] = ddvdx[-1]
#     dvdt[:,1:-1] -= nu*(ddvy+ddvx[:,1:-1])

#     dvdt[:,0] += nu*ddvx[:,0]*ddvy[:,0]
#     dvdt[:,-1] += nu*ddvx[:,-1]*ddvy[:,-1]
    
    dvdt[0,:] = zero
    dvdt[-1,:] = zero # reflective boundaries
    dvdt[:,0] = zero
    dvdt[:,-1] = zero # reflective boundaries
    return dvdt


# In[11]:


@nb.jit(**argfast)
def dudt_advection(h, n, f, u, v, dx, dy, out,vn=None)  :
    p5 = np.float32(0.5)
    if vn is None:
           vn = (v[:,1:]+v[:,:-1])*p5 # n shaped v
    dudt=out
   # advection in x direction
    dudx = d_dx(u, dx)
    dudt[1:-1] -= u[1:-1]*(dudx[1:] + dudx[:-1])*p5 # advection

    # advection in y direction
    duy = xp.empty(u.shape, dtype=u.dtype)
    dudy = d_dy(u, dy)
    duy[:,1:-1] = ( dudy[:,1:] + dudy[:,:-1] ) * p5
    duy[:,0] = dudy[:,0]
    duy[:,-1] = dudy[:, -1]
    dudt[1:-1] -= (vn[1:]+vn[:-1])*p5*duy[1:-1] # advection
#     dudt[0] -= vn[0]*duy[0]
#     dudt[-1] -= vn[-1]*duy[-1] # closest to applicable position
@nb.jit(**argfast)  
def dvdt_advection(h, n, f, u, v, dx, dy, out,vn=None)  :
    if vn is not None:  vn = vn.T
    dudt_advection(h.T, n.T, f.T, v.T, u.T, dy, dx,  out.T, vn)   
    
@nb.jit(**argfast)   # never use parallel if there is a broadcast-- numba bug
def dudt2_advection(i,j,h, n, f, u, v, dx, dy) : 
    p5 = np.float32(0.5)
    last_v_n_i = (v[i-1,j]+v[i-1,j+1])*p5
    v_n = (v[i,j]+v[i,j+1])*p5  # stradles n[i,j]
        # advection
    # malfunctions when i=0 or i=n
    dudx_last = (u[i,j]-u[i-1,j])/dx   # n-centered dx[i-1] is straddled by u[i,j]-u[i-1,j])
    dudx_next = (u[i+1,j]-u[i,j])/dx
    u_du_dx = u[i,j]*(dudx_last+dudx_next)*p5 # back to u-centered
    
    # malfunctions when j=0 or j=m
    dudy_last = (u[i,j]-u[i,j-1])/dy
    dudy_next = (u[i,j+1]-u[i,j])/dy
    v_du_dy= (dudy_last+dudy_next)*p5*(last_v_n_i+v_n)*p5 # yes p5 twice
    
    advection = u_du_dx+v_du_dy
    return -advection

@nb.jit(**argfast)   # never use parallel if there is a broadcast-- numba bug
def dvdt2_advection(i,j,h, n, f, u, v, dx, dy)  :
    return dudt2_advection(j,i,h.T, n.T, f.T, v.T, u.T, dy, dx ) 

@nb.jit(**argfast)   # never use parallel if there is a broadcast-- numba bug
def dvdt2_advection(i,j,h, n, f, u, v, dx, dy) : 
    p5 = np.float32(0.5)
    last_u_n_j = (u[i,j-1]+u[i+1,j-1])*p5
    u_n = (u[i,j]+u[i+1,j])*p5  # stradles n[i,j]
        # advection
    # malfunctions when i=0 or i=n
    dvdy_last = (v[i,j]-v[i,j-1])/dy   # n-centered dx[i-1] is straddled by u[i,j]-u[i-1,j])
    dvdy_next = (v[i,j+1]-v[i,j])/dy
    v_dv_dy = v[i,j]*(dvdy_last+dvdy_next)*p5 # back to u-centered
    
    # malfunctions when j=0 or j=m
    dvdx_last = (v[i,j]-v[i-1,j])/dx
    dvdx_next = (v[i+1,j]-v[i,j])/dx
    u_dv_dx= (dvdx_last+dvdx_next)*p5*(last_u_n_j+u_n)*p5 # yes p5 twice
    
    advection = v_dv_dy+u_dv_dx
    return -advection

@nb.jit(**argfast)   # never use parallel if there is a broadcast-- numba bug
def dudt_grav(h, n, f, u, v, dx, dy,out,g=p.g) : 
   # g = np.float32(9.81)
    dudt = out
    grav = d_dx(n, -dx/g)
    dudt[1:-1] += grav
    
@nb.jit(**argfast)
def dvdt_grav( h, n, f, u, v, dx, dy,out):
    dudt_grav(h.T, n.T, f.T, v.T, u.T, dy, dx , out.T)

@nb.jit(**argfast)
def dvdt_grav(h, n, f, u, v, dx, dy,out,g=p.g) : 
   # g = np.float32(9.81)
    dvdt = out
    grav = d_dy(n, -dy/g)
    dvdt[1:-1] += grav  
    
@nb.jit(**argfast)   # never use parallel if there is a broadcast-- numba bug
def dudt2_grav(i,j,h, n, f, u, v, dx, dy,g=p.g):
# physics constants
   # g = np.float32(9.81) # gravity
    # gravity  (is this wrong in main routine== check indexing)
    # malfunction if i=0 or i=n 
    grav = (n[i-1,j]-n[i,j])*(g)/dx  #n[i-1] and n[i] straddle u[i]
    return grav



@nb.jit(**argfast)   # never use parallel if there is a broadcast-- numba bug
def dvdt2_grav(i,j,h, n, f, u, v, dx, dy):
    return dudt2_grav(j,i,h.T, n.T, f.T, v.T, u.T, dy, dx )

@nb.jit(**argfast)   # never use parallel if there is a broadcast-- numba bug
def dvdt2_grav(i,j,h, n, f, u, v, dx, dy,g=p.g):
# physics constants
   # g = np.float32(9.81) # gravity
    # gravity  (is this wrong in main routine== check indexing)
    # malfunction if i=0 or i=n 
    grav = (n[i,j-1]-n[i,j])*(g)/dy  #n[i-1] and n[i] straddle u[i]
    return grav


#@nb.njit()   # never use parallel if there is a broadcast-- numba bug
@nb.jit(**argfast_noparallel)
def dudt_coriolis(h, n, f, u, v, dx, dy, out,righthand=True,vn=None) : 
    p5 = np.float32(0.5)
    if vn is None:
        vn = (v[:,1:]+v[:,:-1])*p5 # n shaped v
    dudt = out
    # coriolis force
    fn = f         #(f[:,1:]+f[:,:-1])*0.5 # n shaped f
    fvn = (fn*vn) # product of coriolis and y vel.
    if righthand:
        dudt[1:-1] += (fvn[1:]+fvn[:-1])*p5 # coriolis force
    else:
        dudt[1:-1] -= (fvn[1:]+fvn[:-1])*p5 # coriolis force
    dudt[0] += fvn[0]    # Remove?
    dudt[-1] += fvn[-1]  # remove?

#@nb.njit()   # never use parallel if there is a broadcast-- numba bug
@nb.jit(**argfast_noparallel)
def dvdt_coriolis(h, n, f, u, v, dx, dy, out,vn=None):
    if vn is not None:  vn = vn.T
    dudt_coriolis(h.T, n.T, f.T, v.T, u.T, dy, dx,  out.T,righthand=False,vn=vn)   
    
    
@nb.jit(**argfast_noparallel)   # never use parallel if there is a broadcast-- numba bug   
def dudt2_coriolis(i,j,h, n, f, u, v, dx, dy) :
    p5 = np.float32(0.5)
    # first average v along j axis to make it n-centered
    #v_n[i,j] = (v[i,j]+v[i,j+1])*p5  # average neighbors. #### v_n is n-centered between v'000000s
    last_v_n_i = (v[i-1,j]+v[i-1,j+1])*p5
    v_n = (v[i,j]+v[i,j+1])*p5  # stradles n[i,j]

    coriolis_u = (f[i-1,j]*last_v_n_i+f[i,j]*v_n)*p5 # coriolis force F is n-centered
    return coriolis_u

@nb.jit(**argfast_noparallel)   # never use parallel if there is a broadcast-- numba bug
def dvdt2_coriolis(i,j,h, n, f, u, v, dx, dy):
    return -dudt2_coriolis(j,i,h.T, n.T, f.T, v.T, u.T, dy, dx)

@nb.jit(**argfast_noparallel)   # never use parallel if there is a broadcast-- numba bug   
def dvdt2_coriolis(i,j,h, n, f, u, v, dx, dy) :
    p5 = np.float32(0.5)
    # first average v along j axis to make it n-centered
    #v_n[i,j] = (v[i,j]+v[i,j+1])*p5  # average neighbors. #### v_n is n-centered between v'000000s
    last_u_n_i = (u[i,j-1]+u[i+1,j-1])*p5
    u_n = (u[i,j]+u[i+1,j])*p5  # stradles n[i,j]

    coriolis_v = (f[i,j-1]*last_u_n_i+f[i,j]*u_n)*p5 # coriolis force F is n-centered
    return coriolis_v



@nb.jit(**argfast)   # never use parallel if there is a broadcast-- numba bug
def dudt2_drag(i,j,h, n, f, u, v, dx, dy, mu = p.mu) : 
    p5 = np.float32(0.5)
# physics constants
   # mu = np.float32(0.3)
#attenuation by friction
    last_v_n_i = (v[i-1,j]+v[i-1,j+1])*p5
    v_n = (v[i,j]+v[i,j+1])*p5  # stradles n[i,j]
    # this calculation is different that the one used before because it
    # is u-centered directly rather then averaging it.
    # malfuncitons if i=0 or i = n
    last_depth_x = n[i-1,j]+h[i-1,j]  
    depth = n[i,j]+h[i,j]
    depth_u = (depth+last_depth_x)*p5  # straddles u[i]
    v_u =   (last_v_n_i +v_n)*p5  # straddles u[i]
    uu = u[i,j]
    ro = np.float32(1)  # remove
    drag = mu* uu*np.sqrt(uu*uu+ v_u*v_u)/(ro*depth_u)
    return -drag
@nb.jit(**argfast)   # never use parallel if there is a broadcast-- numba bug
def dvdt2_drag(i,j,h, n, f, u, v, dx, dy):
    return dudt2_drag(j,i,h.T, n.T, f.T, v.T, u.T, dy, dx )

@nb.jit(**argfast) 
def dvdt2_drag(i,j,h, n, f, u, v, dx, dy, mu = p.mu) : 
    p5 = np.float32(0.5)
# physics constants
   # mu = np.float32(0.3)
#attenuation by friction
    last_u_n_j = (u[i,j-1]+u[i+1,j-1])*p5
    u_n = (u[i,j]+u[i+1,j])*p5  # stradles n[i,j]
    # this calculation is different that the one used before because it
    # is u-centered directly rather then averaging it.
    # malfuncitons if i=0 or i = n
    last_depth_y = n[i,j-1]+h[i,j-1]  
    depth = n[i,j]+h[i,j]
    depth_v = (depth+last_depth_y)*p5  # straddles u[i]
    u_v =   (last_u_n_j +u_n)*p5  # straddles u[i]
    vv = v[i,j]
    ro = np.float32(1)  # remove
    drag = mu* vv*np.sqrt(vv*vv+ u_v*u_v)/(ro*depth_v)
    return -drag



@nb.jit(**argfast)   # never use parallel if there is a broadcast-- numba bug
def dudt_mydrag(h, n, f, u, v, dx, dy,out): 
    p5=np.float32(0.5)
    mu = np.float32(0.3)
    dudt = out
    #attenuation
    vna = (v[:,1:]+v[:,:-1])*p5
    depth = p5*(h[:-1]+h[1:]+n[:-1]+n[1:])
    v_u = (vna[1:]+vna[:-1])*p5
    attenu = 1/(depth) * mu * u[1:-1] * np.sqrt(u[1:-1]**2 + v_u**2) # attenuation
    dudt[1:-1] -= attenu
    #dudt[0] += zero  
    #dudt[-1] += zero # reflective boundaries
    #dudt[:,0] += zero
    #dudt[:,-1] += zero # reflective boundaries
@nb.jit(**argfast)  # never use parallel if there is a broadcast-- numba bug
def dvdt_mydrag( h, n, f, u, v, dx, dy,out):
    dudt_mydrag(h.T, n.T, f.T, v.T, u.T, dy, dx , out.T)    
#@nb.njit()   # never use parallel if there is a broadcast-- numba bug   
@nb.jit(**argfast)
def dudt_drag(h, n, f, u, v, dx, dy, out,mu=np.float32(p.mu)):
    p5 = np.float32(0.5)
    dudt = out
    #attenuation
    una = (u[1:]+u[:-1]) * p5
    vna = (v[:,1:]+v[:,:-1])*p5
    attenu = 1/(h+n) * mu * una * np.sqrt(una*una + vna*vna) # attenuation
    dudt[1:-1] -= (attenu[1:] + attenu[:-1])*p5
    #dudt[0] = zero
    #dudt[-1] = zero # reflective boundaries
    #dudt[:,0] = zero
    #dudt[:,-1] = zero # reflective boundaries
#@nb.njit()   # never use parallel if there is a broadcast-- numba bug
@nb.jit(**argfast)
def dvdt_drag( h, n, f, u, v, dx, dy,out):
    dudt_drag(h.T, n.T, f.T, v.T, u.T, dy, dx , out.T)

#@nb.njit()   # never use parallel if there is a broadcast-- numba bug       
@nb.jit(**argfast)
def dudt_all(h, n, f, u, v, dx, dy,out,righthand=True)  : 
    p5=np.float32(0.5)
    vn = (v[:,1:]+v[:,:-1])*p5
    dudt_mydrag(h, n, f, u, v, dx, dy, out)
    dudt_advection(h, n, f, u, v, dx, dy,out,vn=vn)
    dudt_grav(h, n, f, u, v, dx, dy,out)
    dudt_coriolis(h, n, f, u, v, dx, dy,out,righthand,vn=vn)
 
# def dudt_all (h, n, f, u, v, dx, dy,out):   
#     fo = [dudt_mydrag,dudt_advection,dudt_grav,dudt_coriolis]
#     for fun in fo:
#         fun(h, n, f, u, v, dx, dy,out)
                   
#@nb.njit()   # never use parallel if there is a broadcast-- numba bug
@nb.jit(**argfast)
def dvdt_all( h, n, f, u, v, dx, dy, out):
    dudt_all(h.T, n.T, f.T, v.T, u.T, dy, dx,out.T,righthand=False) 

# #@nb.njit()   # never use parallel if there is a broadcast-- numba bug       
# def dudt2_all(i,j,h, n, f, u, v, dx, dy)  : 
#     dudt=0.0
#     fn = [dudt2_drag,dudt2_advection,dudt2_grav,dudt2_coriolis]
#     for fun in fn:
#         dudt+=fun(i,j,h, n, f, u, v, dx, dy)
#     return dudt
@nb.jit(**argfast)   # never use parallel if there is a broadcast-- numba bug       
def dudt2_almostall(i,j,h, n, f, u, v, dx, dy)  : 
    dudt=dudt2_drag(i,j,h, n, f, u, v, dx, dy)+        dudt2_advection(i,j,h, n, f, u, v, dx, dy)+        dudt2_grav(i,j,h, n, f, u, v, dx, dy,g=np.float32(9.81))
    return dudt

@nb.jit(**argfast)   # never use parallel if there is a broadcast-- numba bug       
def dvdt2_almostall(i,j,h, n, f, u, v, dx, dy)  : 
    dvdt=dvdt2_drag(i,j,h, n, f, u, v, dx, dy)+        dvdt2_advection(i,j,h, n, f, u, v, dx, dy)+        dvdt2_grav(i,j,h, n, f, u, v, dx, dy,g=np.float32(9.81))
    return dvdt

def dudt2_no_n(i,j,h, n, f, u, v, dx, dy)  : 
    dudt=dudt2_coriolis(i,j,h, n, f, u, v, dx, dy)+        dudt2_advection(i,j,h, n, f, u, v, dx, dy)
    return dudt

@nb.jit(**argfast)   # never use parallel if there is a broadcast-- numba bug       
def dudt2_all(i,j,h, n, f, u, v, dx, dy)  : 
    dudt=dudt2_almostall(i,j,h, n, f, u, v, dx, dy)+          dudt2_coriolis(i,j,h, n, f, u, v, dx, dy)
    return dudt

@nb.jit(**argfast)   # never use parallel if there is a broadcast-- numba bug
def dvdt2_all(i,j,h, n, f, u, v, dx, dy)  : 
    return dudt2_almostall(j,i,h.T, n.T, f.T,  v.T, u.T,dy, dx)-            dudt2_coriolis(j,i,h.T, n.T, f.T,  v.T, u.T,dy, dx)  # check sign

@nb.jit(**argfast)   # never use parallel if there is a broadcast-- numba bug       
def dvdt2_all(i,j,h, n, f, u, v, dx, dy)  : 
    
    dvdt=dvdt2_almostall(i,j,h, n, f, u, v, dx, dy)
    t = dvdt2_coriolis(i,j,h, n, f, u, v, dx, dy)
    return dvdt-t

@nb.jit(**argfast)
def dudt_orig_mimic(h, n, f, u, v, dx, dy,out,righthand=True)  : 
    p5=np.float32(0.5)
    vn = (v[:,1:]+v[:,:-1])*p5
    dudt_drag(h, n, f, u, v, dx, dy, out) # bad drag matches original
    dudt_advection(h, n, f, u, v, dx, dy,out,vn=vn)
    dudt_grav(h, n, f, u, v, dx, dy,out)
    dudt_coriolis(h, n, f, u, v, dx, dy,out,righthand,vn=vn)
 
# def dudt_all (h, n, f, u, v, dx, dy,out):   
#     fo = [dudt_mydrag,dudt_advection,dudt_grav,dudt_coriolis]
#     for fun in fo:
#         fun(h, n, f, u, v, dx, dy,out)
                   
#@nb.njit()   # never use parallel if there is a broadcast-- numba bug
@nb.jit(**argfast)
def dvdt_orig_mimic( h, n, f, u, v, dx, dy, out):
    dudt_orig_mimic(h.T, n.T, f.T, v.T, u.T, dy, dx,out.T,righthand=False) 


# In[12]:


@nb.jit(**argfast)   # never use parallel if there is a broadcast-- numba bug
def dudt2_all_b(i,j,h, n, f, u, v, dx, dy) : 
    p5 = np.float32(0.5)
    last_v_n_i = (v[i-1,j]+v[i-1,j+1])*p5
    v_n = (v[i,j]+v[i,j+1])*p5  # stradles n[i,j]
        # advection
    # malfunctions when i=0 or i=n
    dudx_last = (u[i,j]-u[i-1,j])/dx   # n-centered dx[i-1] is straddled by u[i,j]-u[i-1,j])
    dudx_next = (u[i+1,j]-u[i,j])/dx
    u_du_dx = u[i,j]*(dudx_last+dudx_next)*p5 # back to u-centered
    
    # malfunctions when j=0 or j=m
    dudy_last = (u[i,j]-u[i,j-1])/dy
    dudy_next = (u[i,j+1]-u[i,j])/dy
    v_du_dy= (dudy_last+dudy_next)*p5*(last_v_n_i+v_n)*p5 # yes p5 twice
    
    advection = -(u_du_dx+v_du_dy)


# physics constants
    g = np.float32(9.81) # gravity
    # gravity  (is this wrong in main routine== check indexing)
    # malfunction if i=0 or i=n 
    grav = (n[i-1,j]-n[i,j])*(g)/dx  #n[i-1] and n[i] straddle u[i]

    # first average v along j axis to make it n-centered
    #v_n[i,j] = (v[i,j]+v[i,j+1])*p5  # average neighbors. #### v_n is n-centered between v'000000s
    last_v_n_i = (v[i-1,j]+v[i-1,j+1])*p5
    v_n = (v[i,j]+v[i,j+1])*p5  # stradles n[i,j]

    coriolis_u = (f[i-1,j]*last_v_n_i+f[i,j]*v_n)*p5 # coriolis force F is n-centered


# physics constants
    mu = np.float32(0.3)
#attenuation by friction
    last_v_n_i = (v[i-1,j]+v[i-1,j+1])*p5
    v_n = (v[i,j]+v[i,j+1])*p5  # stradles n[i,j]
    # this calculation is different that the one used before because it
    # is u-centered directly rather then averaging it.
    # malfuncitons if i=0 or i = n
    last_depth_x = n[i-1,j]+h[i-1,j]  
    depth = n[i,j]+h[i,j]
    depth_u = (depth+last_depth_x)*p5  # straddles u[i]
    v_u =   (last_v_n_i +v_n)*p5  # straddles u[i]
    uu = u[i,j]
    ro = np.float32(1)  # remove
    drag = -mu* uu*np.sqrt(uu*uu+ v_u*v_u)/(ro*depth_u)

    return advection+grav+coriolis_u+drag


fn = [dudt2_all,dudt2_drag,dudt2_advection,dudt2_drag,dudt2_grav,dudt2_coriolis]
fo = [dudt_all,dudt_mydrag,dudt_advection,dudt_drag,dudt_grav,dudt_coriolis]
for foo_new,foo_old in zip(fn,fo):
    print (foo_new,foo_old)
    N=6
    M=4
    h = np.asarray( np.float32(2)+np.random.random((N,M)) ,dtype=np.float32)
    n = np.asarray(np.random.random((N,M)) ,dtype=np.float32)
    u = np.asarray(np.random.random((N+1,M)) ,dtype=np.float32)
    v = np.asarray(np.random.random((N,M+1)) ,dtype=np.float32)
    f = np.asarray(np.random.random((N,M)) ,dtype=np.float32)
    dx = np.float32(0.1)
    dy = np.float32(0.2)

    


    out_n = np.asarray(np.random.random((N,M)), dtype=np.float32)
    out_u = np.asarray(np.random.random((N+1,M)) ,dtype=np.float32)
    out_v = np.asarray(np.random.random((N,M+1)) ,dtype=np.float32)
    old_n = np.asarray(np.random.random((N,M))  ,dtype=np.float32)
    old_u = np.asarray(np.zeros((N+1,M)) ,dtype=np.float32)
    old_v = np.asarray(np.random.random((N,M+1)) ,dtype=np.float32)

 

    foo_old(h=h, n=n, f=f, u=u, v=v, dx=dx, dy=dy,out=old_u)
    print (u.shape,old_u.shape)
    def forward2_u_driver (h, n,f, u, v,  dx, dy, out_u):  
        print (out_u.shape)
        for j in nb.prange(1,out_u.shape[1]-1):
            for i in nb.prange(1,out_u.shape[0]-1):  # all but the edges
                out_u[i,j] = foo_new(i=i,j=j,h=h, n=n, f=f, u=u, v=v, dx=dx, dy=dy)
    
            out_u[0,j]=0
    forward2_u_driver (h, n, f,u, v, dx, dy, out_u)        
    print (out_u)
    print(old_u)
    print(old_u[1:-1,1:-1]-out_u[1:-1,1:-1])
orig_u = dudt(h, n, f, u, v, dx, dy, nu=9, mu=p.mu )
old_u = np.asarray(np.zeros((N+1,M)) ,dtype=np.float32)
dudt_orig_mimic (h=h, n=n, f=f, u=u, v=v, dx=dx, dy=dy,out=old_u)
#forward2_u_driver(h=h, n=n, f=f, u=u, v=v, dt=dt, dx=dx,  dy=dy,out_u=out_u)
print ("original")
print(orig_u)
#print ("new2")
#print (out_u)
print("old")
print(old_u)
print ("diff")
print (old_u-orig_u)
fn=[dvdt2_all,dvdt2_drag,dvdt2_advection,dvdt2_drag,dvdt2_grav,dvdt2_coriolis]

fo=[dvdt_all,dvdt_mydrag,dvdt_advection,dvdt_drag,dvdt_grav,dvdt_coriolis]

for foo_new,foo_old in zip(fn,fo):
    print (foo_new,foo_old)
    N=6
    M=4
    h = np.asarray( np.float32(2)+np.random.random((N,M)) ,dtype=np.float32)
    n = np.asarray(np.random.random((N,M)) ,dtype=np.float32)
    u = np.asarray(np.random.random((N+1,M)) ,dtype=np.float32)
    v = np.asarray(np.random.random((N,M+1)) ,dtype=np.float32)
    f = np.asarray(np.random.random((N,M)) ,dtype=np.float32)
    dx = np.float32(0.1)
    dy = np.float32(0.2)

    


    out_n = np.asarray(np.random.random((N,M)), dtype=np.float32)
    out_u = np.asarray(np.random.random((N+1,M)) ,dtype=np.float32)
    out_v = np.asarray(np.random.random((N,M+1)) ,dtype=np.float32)
    old_n = np.asarray(np.random.random((N,M))  ,dtype=np.float32)
    old_u = np.asarray(np.zeros((N+1,M)) ,dtype=np.float32)
    old_v = np.asarray(np.zeros((N,M+1)) ,dtype=np.float32)


   
    nu =np.float32(9)
    
    foo_old(h=h, n=n, f=f, u=u, v=v, dx=dx, dy=dy,out=old_v)
    print (u.shape,old_u.shape)
    
    def forward2_v_driver (h, n, u, v,  dx, dy, out_v):  
       
        for j in nb.prange(1,out_v.shape[1]-1):
            for i in nb.prange(1,out_v.shape[0]-1):  # all but the edges
                out_v[i,j] = foo_new(i=i,j=j,h=h, n=n, f=f, u=u, v=v, dx=dx, dy=dy)
    
            out_v[0,j]=0
    forward2_v_driver (h, n, u, v,  dx, dy, out_v)        
    print (out_v)
    print(old_v)
    print(old_v[1:-1,1:-1]-out_v[1:-1,1:-1])
orig_v = dvdt(h, n, f, u, v, dx, dy, nu, mu=p.mu )

old_v = np.asarray(np.zeros((N,M+1)) ,dtype=np.float32)
dvdt_orig_mimic (h=h, n=n, f=f, u=u, v=v, dx=dx, dy=dy,out=old_v)
#forward2_u_driver(h=h, n=n, f=f, u=u, v=v, dt=dt, dx=dx,  dy=dy,out_u=out_u)
print ("original")
print(orig_v)
#print ("new2")
#print (out_u)
print("old")
print(old_v)
print ("diff")
print ((old_v-orig_v)[1:-1,1:-1])
@cuda.jit('float32(float32,float32,float32[:,:], float32[:,:], float32[:,:],float32[:,:],float32[:,:],float32,float32)')
def j_foo(i,j, h, n, u, v, f, dx, dy):
    k = h[i,j]+n[i,j]
    return k
# In[32]:


from numba import cuda
import math
@cuda.jit('void(float32[:,:], float32[:,:], float32[:,:],float32[:,:],float32[:,:],float32,float32,float32[:,:])')
def cu_u_driver(h, n, u, v, f, dx, dy, out_u):
    """This kernel function will be executed by a thread."""
    i, j  = cuda.grid(2)
    k = np.float32(0.0)
    if (i < out_u.shape[0]-2) and (j < out_u.shape[1]-2):
        if  i>1  and j>1 :
            p5 = np.float32(0.5)
            last_v_n_i = (v[i-1,j]+v[i-1,j+1])*p5
            v_n = (v[i,j]+v[i,j+1])*p5  # stradles n[i,j]
                # advection
            # malfunctions when i=0 or i=n
            dudx_last = (u[i,j]-u[i-1,j])/dx   # n-centered dx[i-1] is straddled by u[i,j]-u[i-1,j])
            dudx_next = (u[i+1,j]-u[i,j])/dx
            u_du_dx = u[i,j]*(dudx_last+dudx_next)*p5 # back to u-centered

            # malfunctions when j=0 or j=m
            dudy_last = (u[i,j]-u[i,j-1])/dy
            dudy_next = (u[i,j+1]-u[i,j])/dy
            v_du_dy= (dudy_last+dudy_next)*p5*(last_v_n_i+v_n)*p5 # yes p5 twice

            advection = -(u_du_dx+v_du_dy)


        # physics constants
            g = np.float32(9.81) # gravity
            # gravity  (is this wrong in main routine== check indexing)
            # malfunction if i=0 or i=n 
            grav = (n[i-1,j]-n[i,j])*(g)/dx  #n[i-1] and n[i] straddle u[i]

            # first average v along j axis to make it n-centered
            #v_n[i,j] = (v[i,j]+v[i,j+1])*p5  # average neighbors. #### v_n is n-centered between v'000000s
            last_v_n_i = (v[i-1,j]+v[i-1,j+1])*p5
            v_n = (v[i,j]+v[i,j+1])*p5  # stradles n[i,j]

            coriolis_u = (f[i-1,j]*last_v_n_i+f[i,j]*v_n)*p5 # coriolis force F is n-centered


        # physics constants
            mu = np.float32(0.3)
        #attenuation by friction
            last_v_n_i = (v[i-1,j]+v[i-1,j+1])*p5
            v_n = (v[i,j]+v[i,j+1])*p5  # stradles n[i,j]
            # this calculation is different that the one used before because it
            # is u-centered directly rather then averaging it.
            # malfuncitons if i=0 or i = n
            last_depth_x = n[i-1,j]+h[i-1,j]  
            depth = n[i,j]+h[i,j]
            depth_u = (depth+last_depth_x)*p5  # straddles u[i]
            v_u =   (last_v_n_i +v_n)*p5  # straddles u[i]
            uu = u[i,j]
            ro = np.float32(1)  # remove
            drag = -mu* uu*math.sqrt(uu*uu+ v_u*v_u)/(ro*depth_u)

            k = grav+advection+coriolis_u+drag
    out_u[i,j]=k    
    cuda.syncthreads()  # maybe not needed


# In[33]:


#cuda.close()


# In[34]:


@nb.jit(**argfast)
def forward2_v_driver (h, n, u, v, f, dx, dy, out_v):  
        for i in nb.prange(1,out_v.shape[0]-1):
            for j in nb.prange(1,out_v.shape[1]-1):  # all but the edges
                out_v[i,j] = dvdt2_all(i=i,j=j,h=h, n=n, f=f, u=u, v=v, dx=dx, dy=dy)
            out_v[i,0]=0

            
@nb.jit(**argfast)
def forward2_v_driver_grid (gi,gj,ni,mj,h, n, u, v, f, dx, dy, out_v,temp):  
        imin = min(n.shape[0]-1,gi)
        imin = max(imin,1)
        imax = min(n.shape[0]-1,gi+ni)
        jmin = min(n.shape[1]-1,gj)
        jmin = max(jmin,1)
        jmax = min(n.shape[1]-1,gj+mj)
        for i in nb.prange(imin,imax):
            out_v[i,0]=0
            for j in nb.prange(jmin,jmax):  # all but the edges
                out_v[i,j] = dvdt2_all(i=i,j=j,h=h, n=n, f=f, u=u, v=v, dx=dx, dy=dy)
            
            
@nb.jit(**argfast)            
def grid_driver_v(h, n, u, v, f, dx, dy, out_v):
    ni=512
    mj=256
    gi_max = np.int(np.floor(n.shape[0]/ni+1))
    gj_max = np.int(np.floor(n.shape[1]/mj+1))
    temp = np.empty((ni,mj),dtype=n.dtype)
    for gi in nb.prange(0,gi_max):
        for gj in nb.prange(0,gj_max):
             forward2_v_driver_grid (gi*ni,gj*mj,ni,mj,h, n, u, v, f, dx, dy, out_v,temp)
            
      
def forward2_u_driver_py (h, n, u, v,f,  dx, dy, out_u):  
        for i in nb.prange(1,out_u.shape[0]-1):
            for j in nb.prange(1,out_u.shape[1]-1):  # all but the edges
                out_u[i,j] = dudt2_all(i=i,j=j,h=h, n=n, f=f, u=u, v=v, dx=dx, dy=dy)
            out_u[i,0]=0        
njit_forward2_u_driver=nb.jit(forward2_u_driver_py,**argfast)

            

forward2_u_driver=njit_forward2_u_driver    

mytime = time.perf_counter
mytime = time.time
            
def testin():
    N=1400
    M=1000
    h = np.asarray( np.float32(2)+np.random.random((N,M)) ,dtype=np.float32)
    n = np.asarray(np.random.random((N,M)) ,dtype=np.float32)
    u = np.asarray(np.random.random((N+1,M)) ,dtype=np.float32)
    v = np.asarray(np.random.random((N,M+1)) ,dtype=np.float32)
    f = np.asarray(np.random.random((N,M)) ,dtype=np.float32)
    dx = np.float32(0.1)
    dy = np.float32(0.2)
    #p.g = np.float32(1.0)
    nu= np.float32(1.0)
    


    out_n = np.asarray(np.random.random((N,M)), dtype=np.float32)
    out_u = np.asarray(np.random.random((M,N+1)) ,dtype=np.float32)
    out_v = np.asarray(np.random.random((N,M+1)) ,dtype=np.float32)
    old_n = np.asarray(np.random.random((N,M))  ,dtype=np.float32)
    old_u = np.asarray(np.zeros((N+1,M)) ,dtype=np.float32)
    old_v = np.asarray(np.zeros((N,M+1)) ,dtype=np.float32)

    threadsperblock = (32, 32)
    blockspergrid_x = (u.shape[0] + threadsperblock[0]) // threadsperblock[0]
    blockspergrid_y = (u.shape[1] + threadsperblock[1]) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    print ("here we go")
    print( "blocks per grid", blockspergrid)
    print("threads per block",threadsperblock)
    try:
        h1 = cuda.to_device(h)
        n1 = cuda.to_device(n)
        u1 = cuda.to_device(u)
        v1 = cuda.to_device(v)
        f1 = cuda.to_device(f)
        out_u1 =cuda.to_device(out_u)
        ts = []
        for i in range(10):
            t = mytime()  # time.process_time()
            for j in range(1000):    
                cu_u_driver[blockspergrid,threadsperblock](h1, n1, u1, v1, f1, dx, dy, out_u1)
                cu_u_driver[blockspergrid,threadsperblock](h1, n1, u1, v1, f1, dx, dy, out_u1)
                cu_u_driver[blockspergrid,threadsperblock](h1, n1, u1, v1, f1, dx, dy, out_u1)
                cu_u_driver[blockspergrid,threadsperblock](h1, n1, u1, v1, f1, dx, dy, out_u1)
                cu_u_driver[blockspergrid,threadsperblock](h1, n1, u1, v1, f1, dx, dy, out_u1)
                cu_u_driver[blockspergrid,threadsperblock](h1, n1, u1, v1, f1, dx, dy, out_u1)
                cu_u_driver[blockspergrid,threadsperblock](h1, n1, u1, v1, f1, dx, dy, out_u1)
                cu_u_driver[blockspergrid,threadsperblock](h1, n1, u1, v1, f1, dx, dy, out_u1)
                cu_u_driver[blockspergrid,threadsperblock](h1, n1, u1, v1, f1, dx, dy, out_u1)
                cu_u_driver[blockspergrid,threadsperblock](h1, n1, u1, v1, f1, dx, dy, out_u1)
                cu_u_driver[blockspergrid,threadsperblock](h1, n1, u1, v1, f1, dx, dy, out_u1)
                cu_u_driver[blockspergrid,threadsperblock](h1, n1, u1, v1, f1, dx, dy, out_u1)
            t2 =  mytime()  # time.process_time()
            ts.append(t-t2)
        print("cuda") 
        print (np.median(ts),np.min(ts),np.max(ts),np.std(ts))
    finally:
        print ("cuda closer")
        cuda.close()  
    print ("all done")
   # %timeit forward2_u_driver (h, n, u, v,  dx, dy, out_u)
   # %timeit   orig_v = dudt(h, n, f, u, v, dx, dy, nu )
    ts = []
    out_u = np.asarray(np.zeros((N+1,M)) ,dtype=np.float32)
    for i in range(10):
        t = mytime()  # time.process_time()
        for j in range(4):
            forward2_u_driver(h, n, u, v, f, dx, dy, out_u)
            forward2_u_driver(h, n, u, v, f, dx, dy, out_u)
            forward2_u_driver(h, n, u, v, f, dx, dy, out_u)
        t2 =  mytime()  # time.process_time()
        ts.append(t-t2)
    print("numba")
    print (np.median(ts),np.min(ts),np.max(ts),np.std(ts))
    ts = []
    for i in range(10):
        t = mytime()  # time.process_time()
        for j in range(4):    
            orig_v = dudt(h, n, f, u, v, dx, dy, nu)
            orig_v = dudt(h, n, f, u, v, dx, dy, nu )
            orig_v = dudt(h, n, f, u, v, dx, dy, nu )
        t2 =  mytime()  # time.process_time()
        ts.append(t-t2)
    print("orig") 
    print (np.median(ts),np.min(ts),np.max(ts),np.std(ts))

    ts = []
    old_u = np.asarray(np.zeros((N+1,M)) ,dtype=np.float32)
    for i in range(10):
        t = mytime()  # time.process_time()
        for j in range(4):
            dudt_all(h=h, n=n, f=f, u=u, v=v, dx=dx, dy=dy,out=old_u)
            dudt_all(h=h, n=n, f=f, u=u, v=v, dx=dx, dy=dy,out=old_u)
            dudt_all(h=h, n=n, f=f, u=u, v=v, dx=dx, dy=dy,out=old_u)
        t2 =  mytime()  # time.process_time()
        ts.append(t-t2)

    print("dudt_all")         
    print (np.median(ts),np.min(ts),np.max(ts),np.std(ts))        
    out_v = np.asarray(np.zeros((N,M+1)) ,dtype=np.float32)
    ts=[]
    for i in range(10):
        t = mytime()  # time.process_time()
        for j in range(4):
            grid_driver_v(h, n, u, v, f, dx, dy, out_v)
            grid_driver_v(h, n, u, v, f, dx, dy, out_v)
            grid_driver_v(h, n, u, v, f, dx, dy, out_v)
        t2 =  mytime()  # time.process_time()
        ts.append(t-t2)
    print("numba v")
    print (np.median(ts),np.min(ts),np.max(ts),np.std(ts))
    ts=[]
    for i in range(10):
        t = mytime()  # time.process_time()
        for j in range(4):
            forward2_v_driver (h, n, u, v, f,dx, dy, out_v)
            forward2_v_driver (h, n, u, v, f, dx, dy, out_v)
            forward2_v_driver (h, n, u, v, f, dx, dy, out_v)
        t2 =  mytime()  # time.process_time()
        ts.append(t-t2)
    print("numba v")
    print (np.median(ts),np.min(ts),np.max(ts),np.std(ts))
    ts = []
    for i in range(10):
        t = mytime()  # time.process_time()
        for j in range(4):
            orig_v = dvdt(h, n, f, u, v, dx, dy, nu, mu=np.float32(0.3) )
            orig_v = dvdt(h, n, f, u, v, dx, dy, nu, mu=np.float32(0.3) )
            orig_v = dvdt(h, n, f, u, v, dx, dy, nu, mu=np.float32(0.3) )
        t2 =  mytime()  # time.process_time()
        ts.append(t-t2)
    print("orig") 
    print (np.median(ts),np.min(ts),np.max(ts),np.std(ts))

    ts = []
    old_v = np.asarray(np.zeros((N,M+1)) ,dtype=np.float32)
    for i in range(10):
        t = mytime()  # mytime()  # time.process_time()
        for j in range(4):
            dvdt_orig_mimic (h=h, n=n, f=f, u=u, v=v, dx=dx, dy=dy,out=old_v)
            dvdt_orig_mimic (h=h, n=n, f=f, u=u, v=v, dx=dx, dy=dy,out=old_v)
            dvdt_orig_mimic (h=h, n=n, f=f, u=u, v=v, dx=dx, dy=dy,out=old_v)
        t2 =  mytime()  # mytime()  # time.process_time()
        ts.append(t-t2)
    print("dvdt_mimic") 
    print (np.median(ts),np.min(ts),np.max(ts),np.std(ts))
    old_v = np.asarray(np.zeros((N,M+1)) ,dtype=np.float32)
    
    return 

testin()


# In[35]:


import numba as nb
import numpy as np
from numba import cuda
try:
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

    n = 483
    p = 323
    a = np.random.random((n, p)).astype(np.float32)
    b = np.ones((n, p)).astype(np.float32)
    c = np.empty_like(a)

    threadsperblock = (16, 16)
    blockspergrid_x = (n + threadsperblock[0]) // threadsperblock[0]
    blockspergrid_y = (p + threadsperblock[1]) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    print (blockspergrid, threadsperblock)
    cu_add_2d[blockspergrid, threadsperblock](a, b, c)
    get_ipython().run_line_magic('timeit', 'cu_add_2d[blockspergrid, threadsperblock](a, b, c)')
    print (a[-5:, -5:])
    print (b[-5:, -5:])
    print (c[-5:, -5:])
finally: 
    cuda.close()  # ha.  needs this.
    print ("cuda close")

@nb.njit()
def forward2_v_driver (h, n, u, v, f, dx, dy, out_v):  
        for i in nb.prange(1,out_v.shape[0]-1):
            for j in nb.prange(1,out_v.shape[1]-1):  # all but the edges
                out_v[i,j] = dvdt2_all(i=i,j=j,h=h, n=n, f=f, u=u, v=v, dx=dx, dy=dy)
            out_v[i,0]=0
        
      


   

@nb.njit()   # never use parallel if there is a broadcast-- numba bug
def dudt2_grav_cuda(i,j,h, n, f, u, v, dx, dy,grav):
# physics constants
    g = np.float32(9.81) # gravity
    # gravity  (is this wrong in main routine== check indexing)
    # malfunction if i=0 or i=n 
    grav[0] = (n[i-1,j]-n[i,j])*(g)/dx  #n[i-1] and n[i] straddle u[i]

@nb.njit()       
def dudt2_advection_cuda(i,j,h, n, f, u, v, dx, dy,advection) : 
    p5 = np.float32(0.5)
    last_v_n_i = (v[i-1,j]+v[i-1,j+1])*p5
    v_n = (v[i,j]+v[i,j+1])*p5  # stradles n[i,j]
        # advection
    # malfunctions when i=0 or i=n
    dudx_last = (u[i,j]-u[i-1,j])/dx   # n-centered dx[i-1] is straddled by u[i,j]-u[i-1,j])
    dudx_next = (u[i+1,j]-u[i,j])/dx
    u_du_dx = u[i,j]*(dudx_last+dudx_next)*p5 # back to u-centered
    
    # malfunctions when j=0 or j=m
    dudy_last = (u[i,j]-u[i,j-1])/dy
    dudy_next = (u[i,j+1]-u[i,j])/dy
    v_du_dy= (dudy_last+dudy_next)*p5*(last_v_n_i+v_n)*p5 # yes p5 twice
    
    advection[0] = -(u_du_dx+v_du_dy)


@nb.njit()   # never use parallel if there is a broadcast-- numba bug   
def dudt2_coriolis_cuda(i,j,h, n, f, u, v, dx, dy,coriolis_u) :
    p5 = np.float32(0.5)
    # first average v along j axis to make it n-centered
    #v_n[i,j] = (v[i,j]+v[i,j+1])*p5  # average neighbors. #### v_n is n-centered between v'000000s
    last_v_n_i = (v[i-1,j]+v[i-1,j+1])*p5
    v_n = (v[i,j]+v[i,j+1])*p5  # stradles n[i,j]

    coriolis_u[0] = (f[i-1,j]*last_v_n_i+f[i,j]*v_n)*p5 # coriolis force F is n-centered





import math # cuda cant do np.sqrt yet
@nb.njit()   # never use parallel if there is a broadcast-- numba bug
def dudt2_drag_cuda(i,j,h, n, f, u, v, dx, dy, mu ,drag=[]) : 
    p5 = np.float32(0.5)
# physics constants
   # mu = np.float32(0.3)
#attenuation by friction
    last_v_n_i = (v[i-1,j]+v[i-1,j+1])*p5
    v_n = (v[i,j]+v[i,j+1])*p5  # stradles n[i,j]
    # this calculation is different that the one used before because it
    # is u-centered directly rather then averaging it.
    # malfuncitons if i=0 or i = n
    last_depth_x = n[i-1,j]+h[i-1,j]  
    depth = n[i,j]+h[i,j]
    depth_u = (depth+last_depth_x)*p5  # straddles u[i]
    v_u =   (last_v_n_i +v_n)*p5  # straddles u[i]
    uu = u[i,j]
    ro = np.float32(1)  # remove
    drag[0] = -mu* uu*math.sqrt(uu*uu+ v_u*v_u)/(ro*depth_u)

@nb.njit()   # never use parallel if there is a broadcast-- numba bug       
def dudt2_almostall_cuda(i,j,h, n, f, u, v, dx, dy,dudt)  : 
    dudt2_drag_cuda(i,j,h, n, f, u, v, dx, dy,np.float32(1.0),dudt) # fix mu!
    t = dudt[0]
    dudt2_advection_cuda(i,j,h, n, f, u, v, dx, dy,dudt)
    t+= dudt[0]
    dudt2_grav_cuda(i,j,h, n, f, u, v, dx, dy,dudt)
    dudt[0]= t+dudt[0]

@nb.njit()   # never use parallel if there is a broadcast-- numba bug       
def dudt2_all_cuda(i,j,h, n, f, u, v, dx, dy,dudt)  : 
    dudt2_almostall_cuda(i,j,h, n, f, u, v, dx, dy,dudt)  # fix mu
    t = dudt[0]
    dudt2_coriolis_cuda(i,j,h, n, f, u, v, dx, dy,dudt)
    dudt[0] = dudt[0]+t
   # return dudt
    
    
def forward2_u_driver_py (h, n, u, v,f,  dx, dy, out_u,dummy): 
       # print("out_u",out_u.dtype, out_u.shape)
        for i in range(1,out_u.shape[0]-1):
            for j in range(1,out_u.shape[1]-1):  # all but the edges
                dudt2_all_cuda(i,j,h, n, f, u, v, dx, dy,dummy)
                #dudt2_all_cuda(i=i,j=j,h=h, n=n, f=f, u=u, v=v, dx=dx, dy=dy, dudt=dummy)
                out_u[i,j]=dummy[0] #moo
            out_u[i,0]=0        
njit_forward2_u_driver=nb.njit(forward2_u_driver_py)
from numba import cuda
cuda_forward2_u_driver=cuda.jit(forward2_u_driver_py,)
forward2_u_driver=njit_forward2_u_driver  
np.SmartArray = np.asarray
def testin():
    N=1000
    M=500
    h = nb.SmartArray( np.float32(2)+np.random.random((N,M)) ,dtype=np.float32)
    n = nb.SmartArray(np.random.random((N,M)) ,dtype=np.float32)
    u = nb.SmartArray(np.random.random((N+1,M)) ,dtype=np.float32)
    v = nb.SmartArray(np.random.random((N,M+1)) ,dtype=np.float32)
    f = nb.SmartArray(np.random.random((N,M)) ,dtype=np.float32)
    dx = np.float32(0.1)
    dy = np.float32(0.2)
    p.g = np.float32(1.0)
    nu= np.float32(1.0)
    
  #  dx=nb.SmartArray(dx,dtype=np.float32)
  #  dy=nb.SmartArray(dy,dtype=np.float32)


  #  out_n = nb.SmartArray(np.random.random((N,M)), dtype=np.float32)
    out_u = nb.SmartArray(np.random.random((M,N+1)) ,dtype=np.float32)
  #  out_v = nb.SmartArray(np.random.random((N,M+1)) ,dtype=np.float32)
   # old_n = nb.SmartArray(np.random.random((N,M))  ,dtype=np.float32)
  #  old_u = nb.SmartArray(np.zeros((N+1,M)) ,dtype=np.float32)
   # old_v = nb.SmartArray(np.zeros((N,M+1)) ,dtype=np.float32)

    dummy = nb.SmartArray([1],dtype=np.float32)
    dudt2_all(i=2,j=2,h=h, n=n, f=f, u=u, v=v, dx=dx, dy=dy,dudt=dummy)
    
    tdudt2_all = cuda.jit(dudt2_grav_cuda)
    tdudt2_all(2,2,h, n, f, u,v, dx, dy,dummy)
    
    tdudt2_all = cuda.jit(dudt2_advection_cuda)
    tdudt2_all(2,2,h, n, f, u,v, dx, dy,dummy)
    
    tdudt2_all = cuda.jit(dudt2_coriolis_cuda)
    tdudt2_all(2,2,h, n, f, u,v, dx, dy,dummy)
    
    tdudt2_all = cuda.jit(dudt2_drag_cuda)
    tdudt2_all(2,2,h, n, f, u,v, dx, dy,np.float32(p.mu), dummy) # fix
    
    tdudt2_all = cuda.jit(dudt2_almostall_cuda)
    tdudt2_all(2,2,h, n, f, u,v, dx, dy,dummy)

    tdudt2_all = cuda.jit(dudt2_all_cuda)
    tdudt2_all(2,2,h, n, f, u,v, dx, dy,dummy)
    
   # %timeit forward2_u_driver (h, n, u, v,  dx, dy, out_u)
   # %timeit   orig_v = dudt(h, n, f, u, v, dx, dy, nu )
    ts = []
    out_u = nb.SmartArray(np.zeros((N+1,M)) ,dtype=np.float32)
    for i in range(3):
        print ("loop ",i,N,M)
        t = time.perf_counter()  # time.process_time()
        for j in range(2):
            forward2_u_driver(h, n, u, v, f, dx, dy, out_u,dummy)
            forward2_u_driver(h, n, u, v, f, dx, dy, out_u,dummy)
            forward2_u_driver(h, n, u, v, f, dx, dy, out_u,dummy)
        t2 =  time.perf_counter()  # time.process_time()
        print("i loop",i,t-t2)
        ts.append(t-t2)
    print("numba")
    print (np.median(ts),np.min(ts),np.max(ts),np.std(ts))
    ts = []
    old_u = nb.SmartArray(np.zeros((N+1,M)) ,dtype=np.float32)
    for i in range(10):
    
        t = time.perf_counter()  # time.process_time()
        for j in range(2): 
            
            orig_v = dudt(h, n, f, u, v, dx, dy, nu)
            orig_v = dudt(h, n, f, u, v, dx, dy, nu )
            orig_v = dudt(h, n, f, u, v, dx, dy, nu )
        t2 =  time.perf_counter()  # time.process_time()
        ts.append(t-t2)
    print("orig") 
    print (np.median(ts),np.min(ts),np.max(ts),np.std(ts))

    ts = []
    old_u = nb.SmartArray(np.zeros((N+1,M)) ,dtype=np.float32)
    for i in range(10):
        t = time.perf_counter()  # time.process_time()
        for j in range(2):
            dudt_all(h=h, n=n, f=f, u=u, v=v, dx=dx, dy=dy,out=old_u)
            dudt_all(h=h, n=n, f=f, u=u, v=v, dx=dx, dy=dy,out=old_u)
            dudt_all(h=h, n=n, f=f, u=u, v=v, dx=dx, dy=dy,out=old_u)
        t2 =  time.perf_counter()  # time.process_time()
        ts.append(t-t2)

    print("dudt_all")         
    print (np.median(ts),np.min(ts),np.max(ts),np.std(ts))        
    out_v = nb.SmartArray(np.zeros((N,M+1)) ,dtype=np.float32)
    ts=[]
    for i in range(10):
        t = time.perf_counter()  # time.process_time()
        for j in range(4):
            forward2_v_driver (h, n, u, v, f,dx, dy, out_v)
            forward2_v_driver (h, n, u, v, f, dx, dy, out_v)
            forward2_v_driver (h, n, u, v, f, dx, dy, out_v)
        t2 =  time.perf_counter()  # time.process_time()
        ts.append(t-t2)
    print("numba v")
    print (np.median(ts),np.min(ts),np.max(ts),np.std(ts))
    ts = []
    for i in range(10):
        t = time.perf_counter()  # time.process_time()
        for j in range(4):
            orig_v = dvdt(h, n, f, u, v, dx, dy, nu, mu=np.float32(0.3) )
            orig_v = dvdt(h, n, f, u, v, dx, dy, nu, mu=np.float32(0.3) )
            orig_v = dvdt(h, n, f, u, v, dx, dy, nu, mu=np.float32(0.3) )
        t2 =  time.perf_counter()  # time.process_time()
        ts.append(t-t2)
    print("orig") 
    print (np.median(ts),np.min(ts),np.max(ts),np.std(ts))

    ts = []
    old_v = nb.SmartArray(np.zeros((N,M+1)) ,dtype=np.float32)
    for i in range(10):
        t = time.perf_counter()  # time.perf_counter()  # time.process_time()
        for j in range(4):
            dvdt_orig_mimic (h=h, n=n, f=f, u=u, v=v, dx=dx, dy=dy,out=old_v)
            dvdt_orig_mimic (h=h, n=n, f=f, u=u, v=v, dx=dx, dy=dy,out=old_v)
            dvdt_orig_mimic (h=h, n=n, f=f, u=u, v=v, dx=dx, dy=dy,out=old_v)
        t2 =  time.perf_counter()  # time.perf_counter()  # time.process_time()
        ts.append(t-t2)
    print("dvdt_mimic") 
    print (np.median(ts),np.min(ts),np.max(ts),np.std(ts))
    old_v = nb.SmartArray(np.zeros((N,M+1)) ,dtype=np.float32)
    
    return 

testin()

# In[17]:


from numba import cuda, float32

# Controls threads per block and shared memory usage.
# The computation will be done on blocks of TPBxTPB elements.
TPB = 16

@cuda.jit(device=False)
def fast_matmul(A, B, C):
    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x    # blocks per grid

    if x >= C.shape[0] and y >= C.shape[1]:
        # Quit if (x, y) is outside of valid C boundary
        return

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = 0.
    for i in range(bpg):
        # Preload data into shared memory
        sA[tx, ty] = A[x, ty + i * TPB]
        sB[tx, ty] = B[tx + i * TPB, y]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]

        # Wait until all threads finish computing
        cuda.syncthreads()

    C[x, y] = tmp

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
        
A = np.zeros((5,5),dtype=np.float32)  
i = np.arange(A.shape[0])
A[i,i]=2
B=np.zeros_like(A)
C=np.zeros_like(B)
B[i,(i+1)%A.shape[1]] = 3
B
print(np.matmul(B,B.T))
A=nb.SmartArray(A)
B=nb.SmartArray(B)
C=nb.SmartArray(C)
matmul(A,B,C)
C
cuda.close()

    
N=6
M=4
h = np.asarray( np.float32(3)+np.random.random((N,M)) ,dtype=np.float32)
n = np.asarray(np.random.random((N,M)) ,dtype=np.float32)
u = np.asarray(np.random.random((N+1,M)) ,dtype=np.float32)
v = np.asarray(np.random.random((N,M+1)) ,dtype=np.float32)
f = np.asarray(np.random.random((N,M)) ,dtype=np.float32)
dx = np.float32(0.1)
dy = np.float32(0.2)




out_n = np.asarray(np.random.random((N,M)), dtype=np.float32)
out_u = np.asarray(np.random.random((N+1,M)) ,dtype=np.float32)
out_v = np.asarray(np.random.random((N,M+1)) ,dtype=np.float32)
old_n = np.asarray(np.random.random((N,M))  ,dtype=np.float32)
old_u = np.asarray(np.random.random((N+1,M)) ,dtype=np.float32)
old_v = np.asarray(np.random.random((N,M+1)) ,dtype=np.float32)

    
@nb.njit(parallel=True)
def forward2_n_driver (h, n, u, v,  dx, dy, out_n):  

    for i in nb.prange(1,out_n.shape[0]-1):
        for j in nb.prange(1,out_n.shape[1]-1):  # all but the edges
            out_n[i,j] = dndt2(i,j,h, n, u, v, dx, dy) #foo_new(i=i,j=j,h=h, n=n, f=f, u=u, v=v, dx=dx, dy=dy, nu=nu, mu=np.float32(0.3))
        out_n[i,0] =  dndt2_0jedge(i,h, n, u, v, dx,dy)
        out_n[i,-1] = dndt2_mjedge(i,h, n, u, v, dx,dy)

        
    for j in nb.prange(0,out_n.shape[1]):  # edges
        out_n[0,j] =dndt2_0iedge(j,h, n, u, v, dx, dy)
        out_n[-1,j] =dndt2_niedge(j,h, n, u, v, dx, dy)
        

#    for i in nb.prange(0,out_n.shape[0]):  # all but the edges
#        out_n[i,0] =  dndt2_0iedge(i,h.T, n.T,  v.T, u.T,dy,dx)
#        out_n[i,-1] = dndt2_niedge(i,h.T, n.T,  v.T, u.T, dy,dx)          
            
forward2_n_driver (h=h, n=n, u=u, v=v,  dx=dx, dy=dy, out_n=out_n)   
old_n = dndt(h=h, n=n, u=u, v=v, dx=dx, dy=dy)  #foo_old(h=h, n=n, f=f, u=u, v=v, dx=dx, dy=dy, nu=nu, mu=np.float32(0.3))
print (n.shape,old_n.shape)
print (out_n)
print(old_n)
print(old_n[:,:]-out_n[:,:])
print (1/dx,1/dy)
print (((h[2,2]+n[2,2])*u[2,2])/dx)#
#%timeit forward2_n_driver (h, n, u,v, dx, dy, out_n)     
%timeit old_n = dndt(h, n, u, v, dx, dy) 
def ttime():
    N=4000
    M=4500
    h = np.asarray( np.float32(2)+np.random.random((N,M)) ,dtype=np.float32)
    n = np.asarray(np.random.random((N,M)) ,dtype=np.float32)
    u = np.asarray(np.random.random((N+1,M)) ,dtype=np.float32)
    v = np.asarray(np.random.random((N,M+1)) ,dtype=np.float32)
    f = np.asarray(np.random.random((N,M)) ,dtype=np.float32)
    dx = np.float32(0.1)
    dy = np.float32(0.2)
    p.g = np.float32(1.0)



    out_n = np.asarray(np.random.random((N,M)), dtype=np.float32)
    out_u = np.asarray(np.random.random((N+1,M)) ,dtype=np.float32)
    out_v = np.asarray(np.random.random((N,M+1)) ,dtype=np.float32)
    old_n = np.asarray(np.random.random((N,M))  ,dtype=np.float32)
    old_u = np.asarray(np.random.random((N+1,M)) ,dtype=np.float32)
    old_v = np.asarray(np.random.random((N,M+1)) ,dtype=np.float32)

    %timeit forward2_n_driver (h, n, u, v,  dx, dy, out_n)     
    %timeit old_n = dndt(h, n, u, v, dx, dy) 
  #  old_n = dndt(h, n, u, v, dx, dy)   #foo_old(h=h, n=n, f=f, u=u, v=v, dx=dx, dy=dy, nu=nu, mu=np.float32(0.3))
ttime() 
# In[18]:


aa = np.random.randint(0,2,12).reshape((3,4))
aa[aa.nonzero()]=77
nb.jit()
def zeromask(x,m,val=0):
    x[m]=val
    
nb.njit(parallel=True)
def zmp(x,m,val=0):
    for i in nb.prange(m[0].size):
        x[m[0][i],m[1][i]] = val
    

mm = aa.nonzero()
zeromask(aa,mm,33)
#%timeit zeromask(aa,mm,33)

#zmp = nb.njit(zeromask)
zmp(aa,mm,44)
#%timeit zmp(aa,mm,44)

print(aa)

bb = np.greater(aa,1)
print (bb)
mm = bb.nonzero()
print (mm)

del aa,mm


# In[19]:


@nb.jit()  # cannot njit this, because slicing by indexes and array masks not supported in numba
def land(h, u, v, coastx): # how to handle land/above water area
    #boundaries / land
#     coastx = xp.less(h, thresh) # start a little farther than the coast so H+n is never less than zero
    zero = np.float32(0)
    u[1:][coastx] = zero
    u[:-1][coastx] = zero # set vel. on either side of land to zero, makes reflective
    v[:,1:][coastx] = zero
    v[:,:-1][coastx] = zero
    return (u, v)  # remove this ########

@nb.njit()
def border(n, u, v, margwidth=15, alph=0.95):
    alph = np.float32(alph)
    n[0:margwidth] *= alph
    n[-1:-margwidth-1:-1] *= alph
    n[:,0:margwidth] *= alph
    n[:,-1:-margwidth-1:-1] *= alph
    
    u[0:margwidth] *= alph
    u[-1:-margwidth-1:-1] *= alph
    u[:,0:margwidth] *= alph
    u[:,-1:-margwidth-1:-1] *= alph
    
    v[0:margwidth] *= alph
    v[-1:-margwidth-1:-1] *= alph
    v[:,0:margwidth] *= alph
    v[:,-1:-margwidth-1:-1] *= alph
    return n, u, v  # remove this ########


# In[20]:



def forward(h, n, u, v, f, dt, dx, dy, nu, doland, coastx, beta=0, mu=0.3): # forward euler and forward/backward timestep
    """
        beta = 0 forward euler timestep
        beta = 1 forward-backward timestep
    """
    beta = np.float32(beta)
    mu = np.float32(mu)
    
    n1 = n + ( dndt(h, n, u, v, dx, dy) )*dt
    u1 = u + ( beta*dudt(h, n1, f, u, v, dx, dy, mu) +  (one-beta)*dudt(h, n, f, u, v, dx, dy, mu) )*dt
    v1 = v + ( beta*dvdt(h, n1, f, u, v, dx, dy, mu) +  (one-beta)*dvdt(h, n, f, u, v, dx, dy, mu) )*dt
    
    u1, v1 = doland(h, u1, v1, coastx) # handle any land in the simulation
    n1, u1, v1 = border(n1, u1, v1, 15, 0.95)
    return n1, u1, v1

def fbfeedback(h, n, u, v, f, dt, dx, dy, nu, doland, coastx, mu=0.3,beta=1/3, eps=2/3,du=None,dv=None,dn=None ): # forward backward feedback timestep
    p5=np.float32(0.5)
    beta = np.float32(beta)
    eps = np.float32(eps)
    mu = np.float32(mu)
    n1g, u1g, v1g = forward(h, n, u, v, f, dt, dx, dy, nu, doland, coastx, beta=beta, mu=mu) # forward-backward first guess
    #feedback on guess
    
    n1 = n + p5*(dndt(h, n1g, u1g, v1g, dx, dy) + dndt(h, n, u, v, dx, dy))*dt
    u1 = u + p5*(eps*dudt(h, n1, f, u, v, dx, dy, nu, mu)+(one-eps)*dudt(h, n1g, f, u1g, v1g, dx, dy, nu, mu)+dudt(h, n, f, u, v, dx, dy, nu, mu))*dt
    v1 = v + p5*(eps*dvdt(h, n1, f, u, v, dx, dy, nu, mu)+(one-eps)*dvdt(h, n1g, f, u1g, v1g, dx, dy, nu, mu)+dvdt(h, n, f, u, v, dx, dy, nu, mu))*dt
    
    u1, v1 = doland(h, u1, v1, coastx) # how to handle land/coast
    n1, u1, v1 = border(n1, u1, v1, 15, 0.95)
    return n1, u1, v1,None,None,None




def timestep(h, n, u, v, f, dt, dx, dy, nu, coastx, mu=0.3,du=None,dv=None,dn=None): return fbfeedback(h, n, u, v, f, dt, dx, dy, nu, land, coastx, mu=mu,du=du,dv=dv,dn=dn) # switch which integrator/timestep is in use
# def timestep(state): return fbfeedback(state, land) # which integrator/timestep is in use

#


# ![image.png](attachment:image.png)
# https://www.myroms.org/wiki/Time-stepping_Schemes_Review#Forward-Backward_Feedback_.28RK2-FB.29

# In[21]:



@nb.jit()
def generalizedFB(h, n, u, v, f, dt, dx, dy, nu, doland, coastx,mu=0.3,                  du=None,dv=None,dn=None,                  beta=0.281105, eps=0.013, gamma=0.0880                  ): # generalized forward backward feedback timestep
    p5=np.float32(0.5)
    p32 =np.float32(1.5)
    beta = np.float32(beta)
    eps = np.float32(eps)
    gamma = np.float32(gamma)
    mu = np.float32(mu)


    
    #feedback on guess
    dn_m0 = dndt(h, n, u, v, dx, dy)
    if  dn is None: 
        dn = (  dn_m0,dn_m0)
        print ("initialized generalizedFB dn")
    dn_m1,dn_m2 =  dn     # unpack

    n1 = n + ((p32+beta)* dn_m0 - (p5+beta+beta)* dn_m1+ (beta)* dn_m2)*dt
    del  dn_m2

    
    du_p1 =  dudt(h, n1, f, u, v, dx, dy, nu, mu)
    if du is None: 
        du=(du_p1,du_p1,du_p1)
        print ("initialized generalizedFB du")
    du_m0,du_m1,du_m2 = du     # unpack
        
    dv_p1 = dvdt(h, n1, f, u, v, dx, dy, nu, mu)
    if dv is None: 
        dv=(dv_p1,dv_p1,dv_p1)
        print ("initialized generalizedFB dv")
    dv_m0,dv_m1,dv_m2 = dv     # unpack
        
    u1 = u+ ((p5+gamma+eps+eps)*du_p1 +(p5-gamma-gamma-eps-eps-eps)*du_m0 +gamma*du_m1+eps*du_m2)*dt
    del du_m2
    v1 = v+ ((p5+gamma+eps+eps)*dv_p1 +(p5-gamma-gamma-eps-eps-eps)*dv_m0 +gamma*dv_m1+eps*dv_m2)*dt
    del dv_m2

    u1, v1 = doland(h, u1, v1, coastx) # how to handle land/coast
    n1, u1, v1 = border(n1, u1, v1, 15, 0.95)
    
    dv = ( dv_p1,dv_m0,dv_m1)
    du = ( du_p1,du_m0,du_m1)
    dn = (dn_m0,dn_m1)
    return n1, u1, v1, du,dv,dn


# In[22]:


####@nb.njit(parallel=True)
def timestep2(h, n, u, v, f, dt, dx, dy, nu, coastx, mu=0.3,du=None,dv=None,dn=None): return            generalizedFB(h, n, u, v, f, dt, dx, dy, nu, land, coastx,mu=mu,                 du=du,dv=dv,dn=dn,                 beta=0.281105, eps=0.013, gamma=0.0880                 ) # switch which integrator/timestep is in use
# def timestep(state): return fbfeedback(state, land) # which integrator/timestep is in use


# ![image.png](attachment:image.png)
# 
# https://www.myroms.org/wiki/Time-stepping_Schemes_Review#Forward-Backward_Feedback_.28RK2-FB.29

# In[23]:


def simulate(initstate, t, mu=0.3, colormap='seismic'): # gives surface height array of the system after evert dt
    """
        evolve shallow water system from initstate over t seconds
        returns:
            animframes (numpy array of matplotlib imshows) numpy array,
            maxn (the maximum value of n over the duration at each point) numpy array,
            minn (the minimum value of n over the duration at each point) numpy array,
            timemax (the number of seconds until the maximum height at each point) numpy array
    """
    h, n, u, v, f, dx, dy, dt = [qp.asnumpy(initstate.__dict__[k]) for k in ('h', 'n', 'u', 'v', 'coriolis', 'dx', 'dy', 'dt')]#h, state.n, state.u, state.v, state.dx, state.dy, state.dt
    nu = (dx+dy)/1000
    #     state = initstate
    mmax = np.max(np.abs(n))
    landthresh = 1.5*np.max(n) # threshhold for when sea ends and land begins
    itrs = int(np.ceil(t/dt))
    
#     nshdf = HDFStore('etaevolvestorage')
    
    assert (dt >= 0), 'negative dt!' # dont try if timstep is zero or negative
    
    ns = np.memmap('etaevolvedata', dtype='float32', mode='w+', shape=(itrs,)+n.shape)
    #np.zeros((itrs,)+n.shape, dtype=n.dtype)
#     animframes = np.array([(plt.imshow(qp.asnumpy(n), vmin=-mmax, vmax=mmax, cmap=colormap),)])
    maxn = np.zeros(n.shape, dtype=n.dtype) # max height in that area
    minn = np.zeros(n.shape, dtype=n.dtype) # minimum height that was at each point
    timemax = np.zeros(n.shape, dtype=n.dtype) # when the maximum height occured

    # make a mask of land
#    coastx = xp.less(h, landthresh) # where the reflective condition is enforced on the coast
    # make a list of land locations (i,j)
    coastx = xp.greater(h,landthresh).nonzero()
 #   try:
    dv = du = dn = None
    for itr in range(itrs):# iterate for the given number of iterations
        
        n, u, v,du,dv,dn = timestep(h, n, u, v, f, dt, dx, dy, nu, coastx, mu=mu,du=du,dv=dv,dn=dn) # pushes n, u, v one step into the future
        ns[itr] = n
       
#             nshdf.put(str(itr), DataFrame(n)) # save n value to nshdf with key the iteration number
#             animframes = np.append(animframes, (plt.imshow(qp.asnumpy(n), vmin=-mmax, vmax=mmax, cmap=colormap),))
#             np.save('etaevolvestorage', ns)
        maxn = np.max((n, maxn), axis=0) # record new maxes if they are greater than previous records
        minn = np.min((n, minn), axis=0)
        timemax[np.greater(n, maxn)] = itr*dt
#             if (itr%ff = 0):
        
 #   except:
#        print('timestep #:' + str(itr))
#        ns.flush()
#         return animframes, maxn, minn, timemax
#        raise Exception('error occured in simulate, printed timestep#')
    ns.flush()
#     nshdf.close()
    print('simulation complete')
    return ns, maxn, minn, timemax # return surface height through time and maximum heights


# In[24]:


class wonk():

        dur = 500 # duration of period to calculate speed over
        size = (10, 1000) # grid squares (dx's)
        dx = np.single(100, dtype=np.float32) # meters
        dy = np.single(100, dtype=np.float32)
        lat = np.linspace(0, 0, size[0]) # physical location the simulation is over
        lon = np.linspace(0, 0 , size[1])
        h = np.float32(100)*np.ones(size,dtype=np.float32)
        n = 1*lingauss(size, 1/4, 1) # intial condition single wave in the center
        u = np.zeros((size[0]+1, size[1]+0)) # x vel array
        v = np.zeros((size[0]+0, size[1]+1)) # y vel array
        margin = 0.1 # error margin of test
W=wonk
testStart = State(W.h, W.n, W.u, W.v, W.dx, W.dy, W.lat, W.lon)
simdata = simulate(testStart, W.dur)


# # verification

# In[ ]:


#wavespeed and differential tests
import unittest
fooo = []
class testWaveSpeed(unittest.TestCase): # tests if the wave speed is correct
    def setUp(self):
        self.dur = 500 # duration of period to calculate speed over
        self.size = (10, 1000) # grid squares (dx's)
        self.dx = np.single(100, dtype=np.float32) # meters
        self.dy = np.single(100, dtype=np.float32)
        self.lat = np.linspace(0, 0, self.size[0]) # physical location the simulation is over
        self.lon = np.linspace(0, 0 , self.size[1])
        self.h = np.float32(100)
        self.n = 1*lingauss(self.size, 1/4, 1) # intial condition single wave in the center
        self.u = np.zeros((self.size[0]+1, self.size[1]+0)) # x vel array
        self.v = np.zeros((self.size[0]+0, self.size[1]+1)) # y vel array
        self.margin = 0.1 # error margin of test
    def calcWaveSpeed(self, ar1, ar2, Dt): # calculat how fast the wave is propagating out
        midstrip1 = ar1[int(ar1.shape[0]/2),int(ar1.shape[1]/2):]
        midstrip2 = ar2[int(ar1.shape[0]/2),int(ar2.shape[1]/2):]
        peakloc1 = np.argmax(midstrip1)
        peakloc2 = np.argmax(midstrip2)
        plt.figure(1)
        plt.clf()
        plt.plot(midstrip1)
        plt.plot(midstrip2)
        plt.show()
        speed = (peakloc2 - peakloc1)*self.dy/Dt
        return speed
    def calcExactWaveSpeed(self): # approximently how fast the wave should be propagating outwards
        ws = np.sqrt(p.g*np.average(self.h))
        return ws
    def test_wavespeed(self): # test if the expected and calculated wave speeds line up approcimently
        self.testStart = State(self.h, self.n, self.u, self.v, self.dx, self.dy, self.lat, self.lon)
        self.simdata = simulate(self.testStart, self.dur)
#         self.testFrames, self.testmax, self.testmin = self.simdata[:3]
        self.testFrames = self.simdata[0]
        self.testEndN = self.testFrames[-1]
        calcedws = self.calcWaveSpeed( self.testStart.n, self.testEndN, self.dur )
        exactws = self.calcExactWaveSpeed()
        err = (calcedws - exactws)/exactws
        print(err, self.margin)
        assert(abs(err) < self.margin) # error margin
    def tearDown(self):
        del(self.dur)
        del(self.dx)
        del(self.dy)
        del(self.lat)
        del(self.lon)
        del(self.size)
        del(self.h)
        del(self.n)
        del(self.u)
        del(self.v)

class testdifferential(unittest.TestCase): # differental function test (d_dx)
    def setUp(self):
        self.a = np.arange(144) # test input
        self.a = self.a.reshape(12, 12) # make into 2d array
        self.ddthreshold = 1E-16
    def test_ddx(self):
        da = d_dx(self.a, 1)
        diff = np.abs(da[1:-1] - np.mean(da[1:-1]))
        maxdiff = np.max(diff)
        self.assertTrue(np.all(np.abs(da[-1:1] < self.ddthreshold)),"expected zero along borders")
        self.assertTrue(np.all(diff < self.ddthreshold),"Expected constant d_dx less than %f but got %f"%(self.ddthreshold,maxdiff))
    def tearDown(self):
        del(self.a)
        del(self.ddthreshold)

unittest.main(argv=['first-arg-is-ignored'], exit=False)
#You can pass further arguments in the argv list, e.g.
#unittest.main(argv=['ignored', '-v'], exit=False)      
#unittest.main()


# In[ ]:


simpletestcase = {
    'h': 1000*np.ones((100, 100), dtype=np.float),
    'n': planegauss((100, 100), ((-4, 4),(-4,4))),
    'u': np.zeros((101, 100)),
    'v': np.zeros((100, 101)),
    'dx': 100,
    'dy': 100,
    'lat': np.zeros((100,)),
    'lon': np.zeros((100,))
}
simpleState = State(**simpletestcase)
simpleframes, simpleMax = simulate(simpleState, 200)[0:2]#, simpleMax, simpleMin, simpleAT

# fig = plt.figure(23)
# plt.imshow(simpleframes[50])v


# In[ ]:


fig = plt.figure(25)
mmax = np.max(np.abs(simpleframes))/2
simpleart = [(plt.imshow(simplef, vmin=-mmax, vmax=mmax, cmap='nipy_spectral'),) for simplef in simpleframes[::5]]
anim = animation.ArtistAnimation(fig, simpleart, interval=50, blit=True, repeat_delay=200)
plt.colorbar()
plt.show()
fig = plt.figure(27)
plt.imshow(simpleMax)


# In[ ]:


friccase = {
    'h': 1000,#*np.ones((100, 100), dtype=np.float),
    'n': planegauss((100, 100)),
    'u': np.zeros((101, 100)),
    'v': np.zeros((100, 101)),
    'dx': 100,
    'dy': 100,
    'lat': np.zeros((100,)),
    'lon': np.zeros((100,))
}
fricState = State(**friccase)
fricframes = simulate(fricState, 100, 100)[0]#, simpleMax, simpleMin, simpleAT
nofricframes = simulate(fricState, 100, 100, 0.0)[0]
# fig = plt.figure(23)
# plt.imshow(simpleframes[50])v


# In[ ]:


fig = plt.figure(25)
fricdif = (fricframes-nofricframes)/(np.abs(fricframes)+np.abs(nofricframes)+1E-10)
fricart = [(plt.imshow(fricf, vmin=np.min(fricdif), vmax=np.max(fricdif)),) for fricf in fricdif]
anim = animation.ArtistAnimation(fig, fricart, interval=50, blit=True, repeat_delay=0)
plt.colorbar()
plt.show()
print(np.mean(fricdif))


# # indonesian tsunami

# In[ ]:



class indone2004():
    event = {
        'lat': 3.316,
        'lon': 95.854
    } # source of the tsunami
    
    dlat = 111000 # latitude degree to meters
    psize = (dlat*30*np.cos(22.5*np.pi/180), dlat*15) # physical size of area
    size = (2500, 1250) # grid squares (dx) # lat, lon

    dx = np.single(psize[0]/size[0], dtype=np.float32) # meters
    dy = np.single(psize[1]/size[1], dtype=np.float32) # meters

    
    bath = nc.Dataset('../data/bathymetry.nc','r')
    
    rxy = (8, 16)
    lat = bath.variables['lat'][:]#[latin]
    lon = bath.variables['lon'][:]
    latr = (np.abs(lat-event['lat']+rxy[1]).argmin(), np.abs(lat-event['lat']-rxy[1]).argmin())
    lonr = (np.abs(lon-event['lon']+rxy[0]).argmin(), np.abs(lon-event['lon']-rxy[0]).argmin())
    latin = np.linspace(latr[0], latr[1], size[0], dtype=int)
    lonin = np.linspace(lonr[0], lonr[1], size[1], dtype=int) # indexes of the bathymetry dataset we need
    lat = bath.variables['lat'][latin]
    lon = bath.variables['lon'][lonin]
    h = np.asarray(-bath.variables['elevation'][latin, lonin], dtype=np.float32)

    n = np.zeros(size)

    evinlat = np.argmin(np.abs(lat - event['lat']))
    evinlon = np.argmin(np.abs(lon - event['lon'])) # the index of the closest value to the correct longitude
    rady = 1+2*(int(25000/dy)//2) # number of indicies across the disturbance is
    radx = 1+2*(int(25000/dx)//2) # modified to be odd, so a point lands on the max of the gaussian
#     evpatch = \

#     evpatch = \
    n[evinlat-rady:evinlat+rady, evinlon-radx:evinlon+radx] =     50*planegauss((2*rady, 2*radx))

    u = np.zeros((size[0]+1, size[1]+0))
    v = np.zeros((size[0]+0, size[1]+1))

indonesia = State(*(props(vars(indone2004))))


# In[ ]:


seasurface = qp.asnumpy(indonesia.n)#np.empty(qp.asnumpy(indonesia.n).shape)
# seasurface[::-1] = qp.asnumpy(indonesia.n)
bathymetry = -qp.asnumpy(indonesia.h)


plt.figure(116)
plt.title('initial conditions of indonesia simulation')

# plt.subplot(121)
a1 = plt.imshow(seasurface, cmap='seismic', vmin=-np.max(seasurface), vmax=np.max(seasurface))
tt1 = plt.title('initial sea surface height')
cb1 = plt.colorbar()
cb1.set_label('sea surface height (m)')

# plt.subplot(222)
a2 = plt.contour(-bathymetry, cmap='Greys')
tt2 = plt.title('bathymetry')
# cb2 = plt.colorbar()
# cb2.set_label('ocean depth (m)')

# plt.subplot(122)
# # a3 = vect()
# tt3 = plt.title('inital velocity (m/s)')


# In[ ]:



indosim = simulate(indonesia, 250)
indot = indosim[0]
maxindo = indosim[1]
minindo = indosim[2]
tmindo = indosim[3]

masq = np.zeros(qp.asnumpy(indonesia.h).shape, dtype=qp.asnumpy(indonesia.h).dtype)
runuplocs = np.array([(5.251, 95.253), (5.452, 95.242), (5.389, 95.960), (2.575, 96.269), (4.208, 96.040)])
radx = masq.shape[0]//100
rady = masq.shape[1]//100
for runuploc in runuplocs:
    arglat, arglon = np.argmin(np.abs(indonesia.lat-runuploc[0])), np.argmin(np.abs(indonesia.lon-runuploc[1])) # gat the index location of this event
    masq[arglat-radx:arglat+radx, arglon-rady:arglon+rady] = 1 # make a small blip around the locatoinplt.figure(123)
plt.contour(masq)
plt.show()
# In[ ]:


# shallowWater/data/2004indonesiarunups.txt


# In[ ]:


indof = np.transpose(indot, (0, 1, 2))
print(indof.shape)
#maxt = np.max(indof,axis=(1,2))
#print(maxt.shape)
#imaxt = np.float32(1.0/maxt)
# plt.figure(888)
# plt.semilogy(maxt)
# plt.show()
norm_indof = indof[::50] #*imaxt[:,np.newaxis,np.newaxis]

h = qp.asnumpy(indonesia.h)
# ht = np.transpose(h)
ht = h


# In[ ]:


fig = plt.figure(123)

# plt.subplot(1, 3, 1)
plt.title('movie')
# f = genframes(norm_indof*0.3, frames=np.linspace(0, norm_indof.shape[0]-1, 300, dtype=int))
indoArts = [(plt.imshow(normindoframe, cmap='seismic', vmin=-np.max(normindoframe), vmax=np.max(normindoframe)),) for normindoframe in norm_indof]
indoAnim = animation.ArtistAnimation(fig, indoArts)
cb = plt.colorbar()
cbtt = cb.set_label('sea surface height (m)')


coast = plt.contour(ht, colors='black', levels=1)#, levels=3)
# locmask = plt.contour(masq, colors='green', levels=2)
plt.show()
plt.figure(211)
plt.subplot(1,3,3)
plt.imshow(indof[1,1225:1275,600:650])
plt.subplot(1,3,2)
plt.imshow(indof[-2,1225:1275,600:650])
plt.subplot(1,3,1)
plt.imshow((indof[-2]-indof[1])[1225:1275,600:650])
plt.figure()
plt.show()


# In[ ]:


fig = plt.figure(124)
plt.title('maximum')
plt.imshow(maxindo+1, cmap='seismic', norm=mpl.colors.LogNorm())
# plt.colorbar()
plt.contour(ht-20, colors='black', levels=1)


# In[ ]:


fig = plt.figure(126)
plt.title('minimum')
plt.imshow(1-minindo, cmap='seismic', norm=mpl.colors.LogNorm())
    # plt.colorbar()
plt.contour(ht-20, colors='black', levels=1)

plt.show()


# # comparing indonesia sim to real data

# In[ ]:


import pandas as pd

ff = pd.read_csv('~rrs/shallowWater/data/2004indonesiarunups2.txt',sep='\t')
#with open('~rrs/shallowWater/data/2004indonesiarunups.txt','r') as f:
#     txt = f.read()
ff   


# In[ ]:


# includes the whole world
allruns = pd.read_csv('~rrs/shallowWater/data/al2004indorunups2.txt', sep='\t')
allruns = allruns.dropna(how='any',subset=['Latitude', 'Longitude', 'TTHr', 'TTMin', 'MaxWaterHeight'])
allruns = allruns.loc[:,['Latitude','Longitude','MaxWaterHeight','DistanceFromSource','TTHr','TTMin','Name']]


# In[ ]:


# import data from file
indorut = pd.read_csv('~rrs/shallowWater/data/2004indonesiarunups2.txt',sep='\t')
indoevents = list(indorut.transpose().to_dict().values()) # list of dicts of events' properties
indoevents


# In[ ]:


# sort the events by proximity to the source
# indoevents.sort(key = lambda event: np.sqrt((indone2004.event['lat'] - event['Latitude'])**2 + (indone2004.event['lat'] - event['Longitude'])**2))
indoevents.sort(key = lambda event: event['DistanceFromSource'])


# In[ ]:


# simindomaxh = np.array([], dtype=np.float32) # the list of the maximum height that occured at each location
# for event in indoevents: # get the maximum water height at each location in the sim in the order given to us
# #     print(event)
#     evlat = event['Latitude'] # the latitude of the measurement
#     evlon = event['Longitude'] # "  longitude " "       "
#     argevlat = np.argmin(np.abs(indonesia.lat-evlat)) # the array index with latitude closest to point
#     argevlon = np.argmin(np.abs(indonesia.lon-evlon)) # the array index with longitude closest to point
#     mh = maxindo[argevlat, argevlon] # the maximum height at this point
#     if (argevlat == 0 or argevlon == 0): # if the point is out of the map so it just returns the edge
#         mh = 0 # ignore the value at the edge 
#     simindomaxh = np.append(simindomaxh, mh)

# list of indexes maximum height at the location of each event
# simindomaxh = [np.max(\
#                       maxindo[\
#                               np.argmin(np.abs(indonesia.lon-event['Longitude'])), \
#                               np.argmin(np.abs(indonesia.lat-event['Latitude']))]) \
#                for event in indoevents]
def simindomaxhgenerator():
    for event in indoevents:
        ilon = np.argmin(np.abs(indonesia.lon-event['Longitude']))
        ilat = np.argmin(np.abs(indonesia.lat-event['Latitude']))
        rad = 20
        ilon = np.max((ilon, rad))
        ilat = np.max((ilat, rad))
        
#         print(maxindo.shape, maxindo[ilon-rad:ilon+rad, ilat-rad:ilat+rad].shape, [ilon-rad,ilon+rad, ilat-rad,ilat+rad])
        try:
            mh = np.max(maxindo[ilon-rad:ilon+rad, ilat-rad:ilat+rad])
        except (ValueError): # when the section is entirely off the map so it returns an empty array
            mh = -1
        yield mh

simindomaxh = np.asarray(list(simindomaxhgenerator()))
simindomaxh


# In[ ]:


indomaxheights = np.array([event['MaxWaterHeight'] for event in indoevents]) # the max water heights of each of the events in the order they are listed


# In[ ]:


#doesn't work - times not recorded
# indotimemh = [event['TTHr']*3600+event['TTMin']*60 for event in indoevents] # a 2d map of the time it took to get there
# indotimemh


# In[ ]:


plt.figure(133)
plt.plot([event['DistanceFromSource'] for event in indoevents])
plt.show()


# In[ ]:


plt.figure(236)
datamap = np.greater(simindomaxh, 0)
print(datamap)
plt.title('maximum heights at various locations')

plt.plot(indomaxheights[datamap], simindomaxh[datamap])

# plt.plot(indomaxheights[datamap], label='measurement') # real data of max water heights

# plt.plot(simindomaxh[datamap], label='simulation') # simulation max water heights in the same order

plt.legend()
plt.show()


# # Palu event

# In[ ]:


def paluEvent(klass, elat, elon):
    pev = klass
#     pev = paluClass
    y = np.argmin(np.abs(pev.lat-elat))
    x = np.argmin(np.abs(pev.lon-elon))
    ry = 1+2*(int(25000/pev.dy)//2) # number of indicies across the disturbance is
    rx = 1+2*(int(25000/pev.dx)//2) # modified to be odd, so one point lands on tip og gaussian
    evpatch = pev.n[y-ry:y+ry, x-rx:x+rx].shape
    pev.n[y-ry:y+ry, x-rx:x+rx] = 10*planegauss((evpatch[0], evpatch[1]))
    full = np.sum(pev.n) # the total amount of initialy displaced water in the system
    pev.n[np.less(pev.h, 1.5*np.max(pev.n))] = 0#-pev.n[np.less(pev.h, 1.5*np.max(pev.n))]
    insea = np.sum(pev.n) # real displaced water, off of land
    pev.n *= full/(insea+1.0E-10) # normalize so all waves are equal even if partially on land
    pstate = State(*(props(vars(pev))))
    return pstate
# palu = paluEvent(**{'elat':-0.2, 'elon':120.4}) # palu, indonesia recent tsunami
# palu = State(*(props(vars(ev))))
def peventd(base, elat, elon):
    newn = np.array(dict(base)['n'])
    y = np.argmin(np.abs(base['lat']-elat))
    x = np.argmin(np.abs(base['lon']-elon))
    ry = 1+2*(int(25000/base['dy'])//2) # number of indicies across the disturbance is
    rx = 1+2*(int(25000/base['dx'])//2) # modified to be odd, so one point lands on tip of gaussian
    evpatch = newn[y-ry:y+ry, x-rx:x+rx].shape # mechanism to still work when partially offmap
    newn[y-ry:y+ry, x-rx:x+rx] = 10*planegauss((evpatch[0], evpatch[1]))
    
    full = np.sum(newn) # the total amount of initialy displaced water in the system
    newn[np.less(base['h'], 1.5*np.max(newn))] = 0 #get rid of water on land
    insea = np.sum(newn) # real displaced water, off of land
    newn *= full/(insea+1.0E-10) # normalize so all waves are equal even if partially on land
    return newn


# In[ ]:


palu = {}
latran = (-1.2, 1.2) # latitude range map covers
lonran = (118.2, 121.2) # longitude range map covers

# calculate height of map  11100*lat degrees = meters
# calculate width of map  1 lon degree = cos(lat) lat degrees, *11100 = meters
# use lon degree size of average latitude
realsize = (111000*(latran[1]-latran[0]),               111000*(lonran[1]-lonran[0])                  *np.cos((latran[1]-latran[0])/2))# h, w of map in meters

size = (500, 625) # grid size of the map lat, lon


palu['dx'] = np.single(realsize[1]/size[1], dtype=np.float32)
palu['dy'] = np.single(realsize[0]/size[0], dtype=np.float32)

# read in bathymetry data
bathdata = nc.Dataset('../data/bathymetry.nc','r')
bathlat = bathdata.variables['lat']
bathlon = bathdata.variables['lon']
#calculate indexes of bathymetry dataset we need
bathlatix = np.linspace(np.argmin(np.abs(bathlat[:]-latran[0])),                        np.argmin(np.abs(bathlat[:]-latran[1])),                        size[0], dtype=int)
bathlonix = np.linspace(np.argmin(np.abs(bathlon[:]-lonran[0])),                        np.argmin(np.abs(bathlon[:]-lonran[1])),                        size[1], dtype=int)
# print(bathlatix, bathlonix)
palu['h'] = np.asarray(-bathdata.variables['elevation'][bathlatix, bathlonix])
palu['lat'] = np.asarray(bathlat[bathlatix])
palu['lon'] = np.asarray(bathlon[bathlonix])

palu['n'] = np.zeros(size, dtype=np.float32)
palu['u'] = np.zeros((size[0]+1,size[1]), dtype=np.float32)
palu['v'] = np.zeros((size[0],size[1]+1), dtype=np.float32)
paluState = State(**palu)


# In[ ]:



fig = plt.figure(166)
coast = plt.contour(palu['h'], levels=1, colors='black')
xtixks = plt.xticks(np.linspace(0, palu['h'].shape[1], 5),           np.round(np.linspace(palu['lon'][0], palu['lon'][-1], 5), 3))
yticks = plt.yticks(np.linspace(0, palu['h'].shape[0], 5),           np.round(np.linspace(palu['lat'][0], palu['lat'][-1], 5), 3))


# In[ ]:



eventcount = 6
center = (-0.63,119.75)
events = np.array([center])
rad = 0.4 # latitude degrees
theta = np.pi/2
for ev in range(eventcount):
    events = np.vstack( ( events, (np.sin(theta)*rad+center[0], np.cos(theta)*rad+center[1]) ) )
    theta += np.pi/8
events = events[1:] #get rid of the made up start point
plt.figure(172, figsize=(5, 5))

plt.scatter(events[:,1], events[:,0])
# plt.contour(palu.h, levels=1, colors='black')

plt.ylim(center[0]-1.5*rad, center[0]+1.5*rad)
plt.xlim(center[1]-1.5*rad, center[1]+1.5*rad)


fig = plt.figure(174)
coast = plt.contour(palu['h'], levels=1, colors='black')
xtixks = plt.xticks(np.linspace(0, palu['h'].shape[1], 5),           np.round(np.linspace(palu['lon'][0], palu['lon'][-1], 5), 3))
yticks = plt.yticks(np.linspace(0, palu['h'].shape[0], 5),           np.round(np.linspace(palu['lat'][0], palu['lat'][-1], 5), 3))


# In[ ]:


displays = np.array([])
eventnum = 0
for event in events:
    evinit = dict(palu)
    evinit['n'] = peventd(evinit, *event)
    paluEventState = State(**evinit)#paluEvent(State(**dict(palu)), *event) # generates initial state a the event location
    displays = np.append(displays, {})
    
    displays[eventnum]['initial'] = paluEventState.n
    
    paluEventSim = simulate(paluEventState, 2000)
    print('finished event ' + str(eventnum))
    displays[eventnum]['animation'] = paluEventSim[0]
    displays[eventnum]['max'] = paluEventSim[1]
    displays[eventnum]['min'] = paluEventSim[2]
    eventnum += 1


# In[ ]:


print(displays.shape)

kk = displays[0].keys()
import h5py
hh = dict()

for i in kk:
    hh[i] = []
for i in displays:
    for k in kk:
        hh[k].append(i[k])
with h5py.File("mytestfile1.hdf5", "w") as f :   
    for i in kk:
        if (i != 'animation'):
            f[i]=hh[i]


#for i in displays:
 #   pf.append(i,ignore_index=True)
# 


# In[ ]:


# np.save('palueventsarray', displays)
# OR
file = HDFStore('paluevents.hdf')
# displaysdf = DataFrame(displays)
eventnum = 0
for event in displays:
    
file.append(str(eventnum), DataFrame(event))
# eventnum += 1
file.close()


# In[ ]:


displays2 = {}
import h5py
# read in data to file var
pevhdf = h5py.File("mytestfile1.hdf5", "r")

# plt.figure(52)
# print(list(pevhdf.keys()))
for k in list(pevhdf.keys()): # for every key in the data
    print(k)
#     displays2 = np.append(displays2, {})
    displays2[k] = np.asarray(pevhdf[k])

displays2


# In[ ]:


# olddisplays = displays
displays = np.array([{},{},{},{},{},{}]) # reset old displays array
for ix in range(len(displays)):
    displays[ix]['initial'] = np.array(displays2['initial'][ix])
    displays[ix]['max'] = np.array(displays2['max'][ix])
    displays[ix]['min'] = np.array(displays2['min'][ix])


# In[ ]:


displays.dtype


# In[ ]:


paluxlim = (310, 360)
paluylim = (60, 120)

sulaxlim = (200, 430)
sulaylim = (0, 250)

startfignum = 0
displaynum = 0
initdisps = [displays[k]['initial'] for k in range(len(displays))]
mmax = np.max(np.abs(initdisps))*0.8 # maximum/-min within init disps
# plot and save initials
for display in initdisps:
    fig = plt.figure(startfignum+displaynum)
    
    # show the SSH
    im = plt.imshow(display[sulaylim[1]:sulaylim[0]:-1, sulaxlim[0]:sulaxlim[1]], cmap='seismic', vmin=-mmax, vmax=mmax)
    # show the coastline
    coast = plt.contour(palu['h'][sulaylim[1]:sulaylim[0]:-1, sulaxlim[0]:sulaxlim[1]], levels=1, colors='black')
    
    # turn off tickmarks
    plt.xticks([], [])
    plt.yticks([], [])
    
    # save figure to results folder
    plt.savefig('../results/paluinit' + str(displaynum))
    
    displaynum += 1
cb = plt.colorbar(im) # get a scale for all above

# plot and save maxes
startfignum = 7
displaynum = 0
maxdisps = np.array([displays[k]['max'] for k in range(len(displays))])
mmax = np.max(np.abs(maxdisps[:, paluylim[1]:paluylim[0]:-1, paluxlim[0]:paluxlim[1]])) # maximum/-min within init disps
for display in maxdisps:
    fig = plt.figure(startfignum+displaynum)
    
    # show the SSH
    im = plt.imshow(display[paluylim[1]:paluylim[0]:-1, paluxlim[0]:paluxlim[1]], cmap='nipy_spectral', vmin=0, vmax=mmax)
    # show the coastline
    coast = plt.contour(palu['h'][paluylim[1]:paluylim[0]:-1, paluxlim[0]:paluxlim[1]], levels=1, colors='black')
    
    # turn off tickmarks
    plt.xticks([], [])
    plt.yticks([], [])
    
    # save figure to results folder
    plt.savefig('../results/palumax' + str(displaynum))
    
    displaynum += 1
cb = plt.colorbar(im) # get color scale


# In[ ]:


print(paluState.dx, paluState.dy, paluState.dt)


# In[ ]:


anim = displays[0]['animation']
snapinterval = np.floor( anim.shape[0] / 20 ) # number of timesteps between snapshots
print('frames taken ' + str(snapinterval*paluState.dt) + ' seconds appart')

snaps = np.array([anim[np.int(snapinterval*sn)] for sn in range(10)]) # np array of several 'snapshot' frames in the animation

startfignum = 18
displaynum = 0

mmax = np.max(np.abs(snaps)) # maximum magnitude throughout snapshots

for snapshot in snaps: # display snapshots in seperate figures
    fig = plt.figure(startfignum+displaynum)
    
    # show the SSH
    im = plt.imshow(snapshot[sulaylim[1]:sulaylim[0]:-1, sulaxlim[0]:sulaxlim[1]], cmap='seismic', vmin=-mmax, vmax=mmax)
    # show the coastline
    coast = plt.contour(palu['h'][sulaylim[1]:sulaylim[0]:-1, sulaxlim[0]:sulaxlim[1]], levels=1, colors='black')
    
    # turn off tickmarks
    plt.xticks([], [])
    plt.yticks([], [])
    
    # save figure to results folder
    plt.savefig('../results/palusnap' + str(displaynum))
    
    displaynum += 1


# In[ ]:


height = 2#len(displays[0].keys()-1)
# width = len(displays)
width = 3
fig = plt.figure(figsize=(6,6))

paluxlim = (310, 360)
paluylim = (60, 110)

sulaxlim = (200, 430)
sulaylim = (0, 250)

displaynum = 1
# for display in displays[:3]:
#     for dispty in [display[k] for k in ['initial', 'max']]:
#         plt.subplot(height, width, displaynum)
#         mmax = np.max(np.abs(dispty))
#         plt.imshow(dispty[::-1], cmap='seismic', vmin=-mmax, vmax=mmax)
#         cb = plt.colorbar()
#         cb.set_label('sea surface height (m)')
#         plt.contour(palu['h'][::-1], levels=1, colors='black')
        
#         xtixks = plt.xticks(np.linspace(0, palu['h'].shape[1], 5),\
#            np.round(np.linspace(palu['lon'][0], palu['lon'][-1], 5), 3))
#         yticks = plt.yticks(np.linspace(0, palu['h'].shape[0], 5),\
#            np.round(np.linspace(palu['lat'][0], palu['lat'][-1], 5), 3))
#         displaynum += 1

for display in [displays[k]['initial'] for k in range(4, 6)]:
    plt.subplot(2, 2, displaynum)
    mmax = np.max(np.abs(display))
    im = plt.imshow(display[sulaylim[1]:sulaylim[0]:-1, sulaxlim[0]:sulaxlim[1]], cmap='seismic', vmin=-mmax, vmax=mmax)
#     cb = plt.colorbar()
#     cb.set_label('sea surface height (m)')
    plt.contour(palu['h'][sulaylim[1]:sulaylim[0]:-1, sulaxlim[0]:sulaxlim[1]], levels=1, colors='black')
    plt.xticks([], [])
    plt.yticks([], [])

#     xtixks = plt.xticks(np.linspace(0, palu['h'].shape[1], 3),\
#        np.round(np.linspace(palu['lon'][0], palu['lon'][-1], 5), 3))
#     yticks = plt.yticks(np.linspace(0, palu['h'].shape[0], 3),\
#        np.round(np.linspace(palu['lat'][0], palu['lat'][-1], 5), 3))
    displaynum += 1

# plt.colorbar()

mmax = np.max(np.abs(displays[0]['max']))
    
for display in [displays[k]['max'] for k in range(4,6)]:
#     for disp in display:
    dispty = display[paluylim[1]:paluylim[0]:-1, paluxlim[0]:paluxlim[1]]
    plt.subplot(2, 2, displaynum)
    
    im = plt.imshow(dispty, cmap='nipy_spectral')#, vmin=-mmax, vmax=mmax)
#     cb = plt.colorbar()
#     cb.set_label('sea surface height (m)')
    plt.contour(palu['h'][paluylim[1]:paluylim[0]:-1, paluxlim[0]:paluxlim[1]], levels=1, colors='black')
    plt.xticks([], [])
    plt.yticks([], [])
#     xtixks = plt.xticks(np.linspace(0, paluylim[1]-paluylim[0], 3),\
#        np.round(np.linspace(palu['lon'][0], palu['lon'][-1], 5), 3))
#     yticks = plt.yticks(np.linspace(0, paluxlim[1]-paluxlim[0], 3),\
#        np.round(np.linspace(palu['lat'][0], palu['lat'][-1], 5), 3))
    displaynum += 1
    
# fig.subplots_adjust(right=0.8)
# cbar_ax = fig.add_axes([0.85, 0.1, 0.03, 0.5])
# fig.colorbar(im, cax=cbar_ax)
# cbar_ax.set_label('sea surface height (m)')


# cb = plt.colorbar(im)
# cb.set_label('sea surface height (m)')
plt.tight_layout(rect=[0, 0, 1, 0.8])#0.8, 0.8])


# In[ ]:


fig.savefig('../results/palumaxheights4-6')


# In[ ]:


plt.figure(182)
plt.imshow(displays[1]['max'])


# In[ ]:


plt.figure(184)
arts = [(plt.imshow(n, cmap='seismic', vmin=-mmax, vmax=mmax),) for n in displays[0]['animation']]
animation.artistAnimation(arts)


# In[ ]:



palutr = (0, 0)
palubl = (400, 400) # palu bay section indices
# disptrans = {}
devinit = [k['initial'] for k in displays[:-1]]
devmax = np.array([k['max'] for k in displays[:-1]])#list of max heights in each simulation
devmin = np.array([k['min'] for k in displays[:-1]])
# print(palutr[1],palubl[1])
paldevmin = devmin[:,palutr[0]:palubl[0],palutr[1]:palubl[1]]
paldevmax = devmax[:,palutr[0]:palubl[0],palutr[1]:palubl[1]]
# normalize to max and min of palu bay section only
pmax = np.max([np.abs(paldevmin), np.abs(paldevmax)])

displaynum = 1
plt.figure(176)
for display in displays[:-1]:
    plt.subplot(3, 2, displaynum)
    maxdata = display['max']/2
    plt.imshow(maxdata, cmap='nipy_spectral')#, cmap='seismic', vmin=-pmax, vmax=pmax)
#     plt.colorbar()
    coast = plt.contour(paluEventState.h-16, levels=1, colors='black')
    displaynum +=1

    
displaynum = 1
plt.figure(177)
for display in displays[:-1]:
    plt.subplot(6, 1, displaynum)
    maxdata = display['max']/2
#     plt.imshow(maxdata, cmap='seismic', vmin=-pmax, vmax=pmax)
    plt.plot(maxdata[np.where(paluEventState.h-21)]==0)
#     plt.colorbar()
    plt.ylim(0, 2)
#     coast = plt.contour(paluEventState.h[palutr[0]:palubl[0], palutr[1]:palubl[1]]-16, levels=1, colors='black')
    displaynum +=1
    

# fignum = 178
# displaynum = 0
# for display in displays[:-1]:
# #     print(display.keys())
#     fig = plt.figure(fignum + displaynum)
#     plt.title(str(displaynum))
    
#     ax1 = fig.add_subplot(2, 2, 1) # initial SSH
#     plt.title('initial ssh')
#     initssh = display['initial']
    
#     plt.imshow(initssh, cmap='seismic', vmin=-pmax, vmax=pmax)
#     coast = plt.contour(paluEventState.h-15, levels=1, colors='black')
    
#     ax3 = fig.add_subplot(2, 2, 3)
#     plt.title('max')
#     maxdata = display['max'][palutr[0]:palubl[0], palutr[1]:palubl[1]]/2
#     plt.imshow(maxdata, cmap='seismic', vmin=-pmax, vmax=pmax)
#     plt.colorbar()
#     coast = plt.contour(paluEventState.h[palutr[0]:palubl[0], palutr[1]:palubl[1]]-15 , levels=1, colors='black')
    
#     ax4 = fig.add_subplot(2, 2, 4)
#     plt.title('min')
#     mindata = display['min'][palutr[0]:palubl[0], palutr[1]:palubl[1]]/2
#     plt.imshow(mindata, cmap='seismic', vmin=-pmax, vmax=pmax)
#     plt.colorbar()
#     coast = plt.contour(paluEventState.h[palutr[0]:palubl[0], palutr[1]:palubl[1]]-15, levels=1, colors='black')
    
#     ax2 = fig.add_subplot(2, 2, 2)
#     displaynum += 1
#     fig = plt.figure(fignum + displaynum)
#     plt.title('animation')
#     frames = display['animation']
#     mmax = np.max(np.abs(frames))/2
#     artis = [(plt.imshow(pframe, cmap='seismic', vmin=-pmax, vmax=pmax),) for pframe in frames]
#     anim = animation.ArtistAnimation(fig, artis)
#     coast = plt.contour(paluEventState.h-15, levels=1, colors='black')
#     anim.save('../results/paluev'+str(displaynum)+'.mp4')
    
    
    
#     displaynum += 1


# In[ ]:


testdata = np.empty((500, 500), dtype=np.float32)
testdata2 = np.empty((500, 500), dtype=np.float32)
emptylist = []
emptynp = np.array([testdata])


# In[ ]:


emptynp


# In[ ]:


get_ipython().run_line_magic('timeit', 'emptynp = np.append(testdata, testdata)')


# In[ ]:


get_ipython().run_line_magic('timeit', 'np.max((testdata2, testdata2), axis=0)')


# In[ ]:


memmapar = np.memmap('memmaptest', dtype='float32', mode='w+', shape=(500, 500, 500))
numpyar = np.zeros((500, 500,500), dtype=np.float32)
get_ipython().run_line_magic('timeit', 'memmapar[0] = testdata')
get_ipython().run_line_magic('timeit', 'numpyar[0] = testdata')

@nb.njit()    # never use parallel if there is a broadcast-- numba bug 
def dndt1(h, n, u, v, dx, dy,out) : # for individual vars
    """change in n per timestep, by diff. equations
    u[n+1,m] v[n,m+1] n[n,m] h[n,m]
    out[n,m]  """
    
    for j in nb.prange(n.shape[1]): # note the flipped order of i and j. on purpose
        last = zero  # careful if you change boundary condition
        last_depth = n[0,j]+h[0,j]  # this economy is bad!  screws up parallizing. remove it
        for i in nb.prange(1,n.shape[0]):
            depth = n[i,j]+h[i,j]
            avex = (last_depth + depth)*p5  
            h = u[i,j]*avex 
            out[i-1,j] = (last-h)/dx[i-1,j]  # is this the right dx or do we need to averageit too? ###
            last = h
            last_depth = depth
        out[-1,j]    =  (h)/dx[-1,j]  # is this the right dx?  is the sign correct?
    
    # in the next loop we use += when dealing with out[]
    for i in nb.prange(n.shape[0]): # note the flipped order of i and j. on purpose 
        last = zero  # careful if you change boundary condition
        last_depth = n[i,0]+h[i,0]
        for j in nb.prange(1,n.shape[1]):
            depth = n[i,j]+h[i,j]
            avey = (last_depth+depth)*p5
            h = v[i,j]*avey 
            out[i,j-1] += (last-h)/dy[i,j-1]  # is this the right dx or do we need to averageit too? ###
            last = h
            last_depth=depth
        # this is made possible by flipped order of i and j
        out[-1,j]  +=  h/dy[i,-1]  # is this the right dy one?    sign?
    return out # superfluous since out is passed indef dndt0(h, n, u, v, dx, dy,out,tmp_u,tmp_v,tmp_hn) : # for individual vars
    """change in n per timestep, by diff. equations
    u[n+1,m] v[n,m+1] n[n,m] h[n,m]
    out[n,m]  tmp_u[n+1,m] tmp_v[n,m+1] tmp_hn[n,m]"""
        # could eliminate one of the tmp array by rearranging calculation
#     h, n, u, v, dx, dy = [qp.asnumpy(state.__dict__[k]) for k in ('h', 'n', 'u', 'v', 'dx', 'dy')]
#    hx = xp.empty(u.shape, dtype=n.dtype) # to be x (u) momentum array
#    hy = xp.empty(v.shape, dtype=n.dtype)
    depth = tmp_hn # rename
    depth[:,:] = h # copy
    depth += n
    depth *= p5
    sum_x(depth,hx[1:-1])
    hx[0] = zero#depth[0]#*2-depth[1] # normal flow boundaries/borders
    hx[-1] = zero#depth[-1]#*2-depth[-2] # the water exiting the water on the edge is n+h
    
    sum_y(depth,hy[:,1:-1])
    hy[:,0] = zero#depth[:,0]#*2-depth[:,1]
    hy[:,-1] = zero#depth[:,-1]#*2-depth[:,-2]
    
    hx *= u # height/mass->momentum of water column.
    hy *= v
#     dndt = d_dx(hx, -dx)+d_dy(hy, -dy)#(div(hx, hy, -dx, -dy))
    d_dx(hx,dx,out)
    d_dy(hy,dy,tmp_hn)
    out += tmp_hn
    out *= np.float32(-1)


    def dudt0(h, n, f, u, v, dx, dy, nu, mu=np.float32(0.3), out,tmp_n,tmp_dudx,tmp_u) : # for individual vars
    mu = np.float32(mu)
# def dudt(state):
#     f, n, u, v, dx, dy = [qp.asnumpy(state.__dict__[k]) for k in ('coriolis', 'n', 'u', 'v', 'dx', 'dy')]
#    dudt = xp.empty(u.shape, dtype=u.dtype) # x accel array
    # if dx is an array want to change the next line to avoid copies
   # dudt[1:-1] = d_dx(n, -dx/p.g)  # gravity  
    # change the following if dx is matrix to avoid copies
    d_dx(n,-dx/p.g,out[1:-1])
    dudt = out
#     dudt[0] = grav[0]*2-grav[1] # assume outside map wave continues with constant slope
#     dudt[-1] = grav[-1]*2-grav[-2]
    
    
    # coriolis force
    sum_y(v,tmp_n)
    tmp_n *= p5
    #vn = (v[:,1:]+v[:,:-1])*p5 # n shaped v
    
    #fn = f#(f[:,1:]+f[:,:-1])*0.5 # n shaped f
    #fvn = (fn*vn) # product of coriolis and y vel.
    fvn = tmp_n *= f  # broadcast f if it's 1D
    
    
    sum_x(fvn, tmp_u[1:-1,:])  # could use a different size matrix here.
    tmp_u*= p5
    
    dudt[1:-1,:] +=  tmp_u[1:-1,:] # coriolis force
    dudt[0] += fvn[0]
    dudt[-1] += fvn[-1]
    
    
    # advection
    
    # advection in x direction
   # dudx = d_dx(u, dx)
    d_dx(u,dx,tmp_n)
    sum_x(tmp_n,tmp_u[1:-1,:])
    tmp_u[1:-1]*=p5 
    tmp_u[1:-1]*= u[1:-1]
    dudt[1:-1] -=  tmp_u[1:-1] # advection
#     dudt[0] -= u[0]*dudx[0]
#     dudt[-1] -= u[-1]*dudx[-1]
    
    # advection in y direction
    #duy = xp.empty(u.shape, dtype=u.dtype)
    #dudy = d_dy(u, dy)
    d_dy(u,dy,tmp_u[:,:-1])
    sum_y(tmp_u[:,:-1], tmp_n[:,:,-1])
    tmp_n[:,:,-1] *=p5
    # copy it back in
    duy = tmp_u
    tmp_u[:,1:-1] = tmp_n[:,:,-1]
    
    ######################################################################
    ##  What I need to do is just have 2 or 3 tmp arrays then define views on these
    ##  the arrays will be T[n+1,m+1]
    ##  tmp_n = T[:-1,:-1]
    ##  tmp_u = T[:,:-1]
    ##  tmp_v = T[:-1,:]
    ##  dudy  = T[:,:-2]
    ##  dvdx  = T[:-2,:]
    ################################################################
    #### okay I'm lost here.
    duy[:,0] = dudy[:,0]
    duy[:,-1] = dudy[:, -1]
    dudt[1:-1] -= (vn[1:]+vn[:-1])*p5*duy[1:-1] # advection
#     dudt[0] -= vn[0]*duy[0]
#     dudt[-1] -= vn[-1]*duy[-1] # closest to applicable position
    
    
    #attenuation
    una = (u[1:]+u[:-1]) * p5
    vna = (v[:,1:]+v[:,:-1])*p5
    attenu = 1/(h+n) * mu * una * np.sqrt(una*una + vna*vna) # attenuation
    dudt[1:-1] -= (attenu[1:] + attenu[:-1])*p5
    
    # viscous term
#     nu = np.float32(1000/dx)

#     ddux = d_dx(dudx, dx)
#     dduy = xp.empty(u.shape, dtype=u.dtype)
#     ddudy = d_dy(duy, dy)
#     dduy[:,1:-1] = ( ddudy[:,1:] + ddudy[:,:-1] ) * p5
#     dduy[:,0] = ddudy[:,0]
#     dduy[:,-1] = ddudy[:, -1]
#     dudt[1:-1] -= nu*(ddux+dduy[1:-1])
    
#     dudt[0] += nu*ddux[0]*dduy[0]
#     dudt[-1] += nu*ddux[-1]*dduy[-1]
    
    dudt[0] = zero
    dudt[-1] = zero # reflective boundaries
    dudt[:,0] = zero
    dudt[:,-1] = zero # reflective boundaries
    return ( dudt )

# countx=0


def dudt_grav(h, n, f, u, v, dx, dy,out ) : # for individual vars
    dudt=out # rename
    grav = d_dx(n, -dx/p.g)
    dudt[1:-1] += grav

    
def dudt_coriolis(h, n, f, u, v, dx, dy,out) : # for individual vars   
    # coriolis force
    dudt = out # rename
    vn = (v[:,1:]+v[:,:-1])*p5 # n shaped v
    
    fn = f        #(f[:,1:]+f[:,:-1])*0.5 # n shaped f
    fvn = (fn*vn) # product of coriolis and y vel.
    dudt[1:-1] += (fvn[1:]+fvn[:-1])*p5 # coriolis force
    dudt[0] += fvn[0]
    dudt[-1] += fvn[-1]
    return dudt
    
    # advection
def dudt_advection(h, n, f, u, v, dx, dy, out)  :    
    dudt = out
   # advection in x direction
    dudx = d_dx(u, dx) 
    dudt[1:-1] = u[1:-1]*(dudx[1:] + dudx[:-1])*p5 # advection
    # advection in y direction
    duy = xp.empty(u.shape, dtype=u.dtype)
    dudy = d_dy(u, dy)
    duy[:,1:-1] = ( dudy[:,1:] + dudy[:,:-1] ) * p5
    duy[:,0] = dudy[:,0]
    duy[:,-1] = dudy[:, -1]
    dudt[1:-1] += (vn[1:]+vn[:-1])*p5*duy[1:-1] # advection
    dudt[:,:]=-dudt
#     dudt[0] -= vn[0]*duy[0]
#     dudt[-1] -= vn[-1]*duy[-1] # closest to applicable position


def dudt_drag(h, n, f, u, v, dx, dy, out): 
    dudt = out
    #attenuation
    una = (u[1:]+u[:-1]) * p5
    vna = (v[:,1:]+v[:,:-1])*p5
    attenu = 1/(h+n) * mu * una * np.sqrt(una*una + vna*vna) # attenuation
    dudt[1:-1] -= (attenu[1:] + attenu[:-1])*p5
    
    # viscous term
#     nu = np.float32(1000/dx)

#     ddux = d_dx(dudx, dx)
#     dduy = xp.empty(u.shape, dtype=u.dtype)
#     ddudy = d_dy(duy, dy)
#     dduy[:,1:-1] = ( ddudy[:,1:] + ddudy[:,:-1] ) * p5
#     dduy[:,0] = ddudy[:,0]
#     dduy[:,-1] = ddudy[:, -1]
#     dudt[1:-1] -= nu*(ddux+dduy[1:-1])
    
#     dudt[0] += nu*ddux[0]*dduy[0]
#     dudt[-1] += nu*ddux[-1]*dduy[-1]
    
#    dudt[0] = zero
##    dudt[-1] = zero # reflective boundaries
#    dudt[:,0] = zero
#    dudt[:,-1] = zero # reflective boundaries
  def dudt2_v_n(i,j,h, n, f, u, v, dx, dy, nu, mu=np.float32(0.3)) : # for individual vars
    mu = np.float32(mu)
def dudt2_coriolis(i,j,h, n, f, u, v, dx, dy, nu, mu=np.float32(0.3)) :
    # first average v along j axis to make it n-centered
    #v_n[i,j] = (v[i,j]+v[i,j+1])*p5  # average neighbors. #### v_n is n-centered between v'000000s
    last_v_n_i = (v[i-1,j]+v[i-1,j+1])*p5
    v_n = (v[i,j]+v[i,j+1])*p5  # stradles n[i,j]
  #  next_v_n_i = (v[i+1,j]+v[i+1,j+1])*p5

 
      # coriolis force

    coriolis_u = (f[i-1,j]*last_v_n_i+f[i,j]*v_n)*p5 # coriolis force F is n-centered
    return coriolis_u

def dudt2_grav(i,j,h, n, f, u, v, dx, dy, nu, mu=np.float32(0.3)):
    # gravity  (is this wrong in main routine== check indexing)
    # malfunction if i=0 or i=n 
    grav = (n[i-1,j]-n[i,j])*(p.g)/dx  #n[i-1] and n[i] straddle u[i]
    return grav

def dudt2_advection(i,j,h, n, f, u, v, dx, dy, nu, mu=np.float32(0.3)) :   
    last_v_n_i = (v[i-1,j]+v[i-1,j+1])*p5
    v_n = (v[i,j]+v[i,j+1])*p5  # stradles n[i,j]
        # advection
    # malfunctions when i=0 or i=n
    dudx_last = (u[i,j]-u[i-1,j])/dx   # n-centered dx[i-1] is straddled by u[i,j]-u[i-1,j])
    dudx_next = (u[i+1,j]-u[i,j])/dx
    u_du_dx = u[i,j]*(dudx_last+dudx_next)*p5 # back to u-centered
    
    # malfunctions when j=0 or j=m
    dudy_last = (u[i,j]-u[i,j-1])/dy 
    dudy_next = (u[i,j+1]-u[i,j])/dy 
    v_du_dy= (dudy_last+dudy_next)*p5*(v_n[i-1,j]+v_n[i,j])*p5 # yes p5 twice
    
    advection = u_du_dx+v_du_dy
    return advection


def dudt2_drag(i,j,h, n, f, u, v, dx, dy, nu, mu=np.float32(0.3)) :   
#attenuation by friction

    # this calculation is different that the one used before because it
    # is u-centered directly rather then averaging it.
    # malfuncitons if i=0 or i = n
    last_depth_x = n[i-1,j]+h[i-1,j]  
    depth = n[i,j]+h[i,j]
    depth_u = (depth+last_depth_x)*p5  # straddles u[i]
    v_u =   (last_v_n_i +v_n)*p5  # straddles u[i]
    uu = u[i,j]
    ro = np.float32(1)
    drag = mu* uu*np.sqrt(uu*uu+ v_u*v_u)/(ro*depth_u)

  #  dudt[0] = zero
  #  dudt[-1] = zero # reflective boundaries
  #  dudt[:,0] = zero
  #  dudt[:,-1] = zero # reflective boundaries
    du_dt = -(-coriolis +grav+advection+drag)
    return  du_dt def dvdt2(i,j,h, n, f, u, v, dx, dy, nu, mu=np.float32(0.3)) : # for individval vars
    mu = np.float32(mu)

    # first average v along j axis to make it n-centered
    #v_n[i,j] = (v[i,j]+v[i,j+1])*p5  # average neighbors. #### v_n is n-centered between v'000000s
    last_u_n_j = (u[i,j-1]+u[i+1,j-1])*p5
    u_n = (u[i,j]+u[i+1,j])*p5  # stradles n[i,j]
  #  next_v_n_i = (v[i+1,j]+v[i+1,j+1])*p5


      # coriolis force

    coriolis_v = (f[i,j-1]*last_v_n_j+f[i,j]*v_n)*p5 # coriolis force F is n-centered
    
    # gravity  (is this wrong in main routine== check indexing)
    # malfunction if i=0 or i=n 
    grav = (n[i,j-1]-n[i,j])*(p.g)/dy[i,j]  #n[i-1] and n[i] straddle u[i]
  
    # advection
    # malfunctions when i=0 or i=n
    dvdy_last = (v[i,j]-v[i,j-1])/dy[i,j-1]   # n-centered dy[i-1] is straddled by u[i,j]-u[i-1,j])
    dvdy_next = (v[i,j+1]-v[i,j])/dy[i,j]
    u_dv_dy = v[i,j]*(dvdy_last+dvdy_next)*p5 # back to u-centered
    
    # malfunctions when j=0 or j=m
    dvdx_last = (v[i,j]-v[i-1,j])/dx 
    dvdx_next = (v[i+1,j]-v[i,j])/dx
    v_dv_dx= (dvdx_last+dvdx_next)*p5*(u_n[i,j-1]+u_n[i,j])*p5 # yes p5 twice
    
    advection = u_dv_dy+v_dv_dy
    
    #attenuation by friction

    # this calculation is different that the one used before because it
    # is u-centered directly rather then averaging it.
    # malfuncitons if i=0 or i = n
    last_depth_y = n[i,j-1]+h[i,j-1]  
    depth = n[i,j]+h[i,j]
    depth_u = (depth+last_depth_y)*p5  # straddles u[i]
    u_v =   (last_u_n_j +u_n)*p5  # straddles u[i]
    vv = v[i,j]
    ro = np.float32(1)
    drag = mu* vv*np.sqrt(vv*vv+ u_v*u_v)/(ro*depth_v)

  #  dvdt[0] = zero
  #  dvdt[-1] = zero # reflective boundaries
  #  dvdt[:,0] = zero
  #  dvdt[:,-1] = zero # reflective boundaries
    dv_dt = -(+coriolis +grav+advection+drag)
    return  dv_dt   def dudt1(h, n, f, u, v, dx, dy, nu, mu=np.float32(0.3),out,tmp) : # for individual vars
    mu = np.float32(mu)
  #  dudt = xp.empty(u.shape, dtype=u.dtype) # x accel array
    du_dt = out  # just a name
    v_n = tmp  # just a name
   
    
  
    # first average v along j axis to make it n-centered
    for i in nb.prange(v.shape[0]):  # N   
        for j in nb.prange(u.shape[1]):  #M
            # average along j 
            v_n[i,j] = (v[i,j]+v[i,j+1])*p5  # average neighbors. #### v_n is n-centered between v'000000s
 
    
      # coriolis force
    # now have to average along i to make this u-centered
    for j in nb.prange(u.shape[1]):  #M
        # note the first valid value in du_dt is at i=1 not i=0
        for i in nb.prange(1,v.shape[0]):  # N   
            du_dt[i,j] = (f[i-1,j]*v_n[i-1,j]+f[i,j]*v_n[i,j])*p5 # coriolis force F is n-centered
        du_dt[0,j]= zero # boudary condition    
        du_dt[-1,j]=zero
    # Completed Coriolis force.

    
    
    # add on gravity
    for j in nb.prange(u.shape[1]):  #M
       # du_dt[0,j] += zero  # boundary conditio
        for i in nb.prange(1,v.shape[0]):  # N
            grav = (n[i,j]-n[i-1,j])*(-p.g)/dx[i,j]  
            du_dt[i,j] += grav
       # du_dt[-1,j] += zero # boundary conditio   
    

  
    # advection
    for j in nb.prange(u.shape[1]):  #M
        last_du_dx= ????  #####################################
        for i in nb.prange(v.shape[0]):  # N  
            dudx = (u[i+1,j]-u[i,j])/dx[i,j]   # n-centered
            du_dt[i+1,j] -= u[i+1,j]*(dudx+last_dudx)*p5 # back to u-centered
            last_dudx = dudx

    
    # advection in y direction
      
    for j in nb.prange(1,u.shape[1]-1):  #M        
        for i in nb.prange(v.shape[0]):  # N 
            dudy = (u[i,j]-u[i,j-1])/dy[i,j-1] 
            dudy += (u[i,j+1]-u[i,j])/dy[i,j] # Should we average dy in a square?
            dudy *= p5
            dudy *= (v_n[i-1,j]+v_n[i,j])*p5
            du_dt[i,j]+=dudy
                #dudy is centered on u
        #du_dt[0,j] += 0 # boundary condition
        #du_dt[-1,j] += 0 # boundary condition
    for i in nb.prange(v.shape[0]):  # N 
        du_dt[i,0] += 0 # boundary condition
        du_dt[i,-1] += 0 # boundary condition
    
    
    
    #attenuation by friction
    for j in nb.prange(1,u.shape[1]-1):  #M  
        last_atten = zero  # this breaks parallelism  but may not matter.
        for i in nb.prange(1,v.shape[0]):  # N 
            u_n = (u[i-1,j]+u[i,j])*p5
            s_n = np.sqrt(u_n*u_n + v_n[i-1,j]**2)
            atten = mu*u_n*s_n/(h[i,j]+n[i,j])
            du_dt[i,j] -= (last_atten+atten)*p5
            last_atten=atten
        #du_dt[0,j] -=zero  # no need
        #du_dt[-1,j] -= zero # no need
        
     # at this point it should be that 
     # the boders of du_dt are all zero
  
   # dudt[0] = zero
   # dudt[-1] = zero # reflective boundaries
   # dudt[:,0] = zero
   # dudt[:,-1] = zero # reflective boundaries
    
    
    # viscous term
#     nu = np.float32(1000/dx)

#     ddux = d_dx(dudx, dx)
#     dduy = xp.empty(u.shape, dtype=u.dtype)
#     ddudy = d_dy(duy, dy)
#     dduy[:,1:-1] = ( ddudy[:,1:] + ddudy[:,:-1] ) * p5
#     dduy[:,0] = ddudy[:,0]
#     dduy[:,-1] = ddudy[:, -1]
#     dudt[1:-1] -= nu*(ddux+dduy[1:-1])
    
#     dudt[0] += nu*ddux[0]*dduy[0]
#     dudt[-1] += nu*ddux[-1]*dduy[-1]
    
    dudt[0] = zero
    dudt[-1] = zero # reflective boundaries
    dudt[:,0] = zero
    dudt[:,-1] = zero # reflective boundaries
    return  du_dt  def dudt2(i,j,h, n, f, u, v, dx, dy, nu, mu=np.float32(0.3)) : # for individual vars
    mu = np.float32(mu)

    # first average v along j axis to make it n-centered
    #v_n[i,j] = (v[i,j]+v[i,j+1])*p5  # average neighbors. #### v_n is n-centered between v'000000s
    last_v_n_i = (v[i-1,j]+v[i-1,j+1])*p5
    v_n = (v[i,j]+v[i,j+1])*p5  # stradles n[i,j]
  #  next_v_n_i = (v[i+1,j]+v[i+1,j+1])*p5


      # coriolis force

    coriolis_u = (f[i-1,j]*last_v_n_i+f[i,j]*v_n)*p5 # coriolis force F is n-centered
    
    # gravity  (is this wrong in main routine== check indexing)
    # malfunction if i=0 or i=n 
    grav = (n[i-1,j]-n[i,j])*(p.g)/dx  #n[i-1] and n[i] straddle u[i]
  
    # advection
    # malfunctions when i=0 or i=n
    dudx_last = (u[i,j]-u[i-1,j])/dx   # n-centered dx[i-1] is straddled by u[i,j]-u[i-1,j])
    dudx_next = (u[i+1,j]-u[i,j])/dx
    u_du_dx = u[i,j]*(dudx_last+dudx_next)*p5 # back to u-centered
    
    # malfunctions when j=0 or j=m
    dudy_last = (u[i,j]-u[i,j-1])/dy 
    dudy_next = (u[i,j+1]-u[i,j])/dy 
    v_du_dy= (dudy_last+dudy_next)*p5*(v_n[i-1,j]+v_n[i,j])*p5 # yes p5 twice
    
    advection = u_du_dx+v_du_dy
    
    #attenuation by friction

    # this calculation is different that the one used before because it
    # is u-centered directly rather then averaging it.
    # malfuncitons if i=0 or i = n
    last_depth_x = n[i-1,j]+h[i-1,j]  
    depth = n[i,j]+h[i,j]
    depth_u = (depth+last_depth_x)*p5  # straddles u[i]
    v_u =   (last_v_n_i +v_n)*p5  # straddles u[i]
    uu = u[i,j]
    ro = np.float32(1)
    drag = mu* uu*np.sqrt(uu*uu+ v_u*v_u)/(ro*depth_u)

  #  dudt[0] = zero
  #  dudt[-1] = zero # reflective boundaries
  #  dudt[:,0] = zero
  #  dudt[:,-1] = zero # reflective boundaries
    du_dt = -(-coriolis +grav+advection+drag)
    return  du_dt 

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

    
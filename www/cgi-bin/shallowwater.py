#!/usr/bin/python
import cgi
import numpy as np
import numba as nb
# from numba import cuda
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
from math import sqrt
mysqrt= sqrt # numba replacement for np.sqrt

# common constants: pre-cast to float32 form to speed up calculations.
#    convenient to make globals
zero = np.float32(0)
p5 = np.float32(0.5)
one = np.float32(1)
two = np.float32(2)

# derivative in x
@nb.njit(fastmath=True,parallel=True)
def d_dx(a, dx):
    return ( a[1:] - a[:-1] )*(np.float32(1)/dx) # ddx

# derivative in y
@nb.njit(fastmath=True,parallel=True)
def d_dy(a, dy):
    return ( a[:,1:] - a[:,:-1] )*(np.float32(1)/dy)

# create a simple 1d gaussian disturbance
def lingauss(shape, w, cx = 0, cy = 0, theta = 0, cutoff = 0.05, norm = False):#, win = (-2, 2)):
    """returns a 1d gaussian on a 2d array of shape 'shape'"""
    x = np.arange(0, shape[0])#linspace( win[0], win[1], shape[0] )
    y = np.arange(0, shape[1])#linspace( win[0], win[1], shape[1] )
    xx, yy = np.meshgrid(x, y, indexing='ij')
    xy = np.cos(theta)*(xx-cx) + np.sin(theta)*(yy-cy) # lin comb of x, y, to rotate gaussian
    h = np.exp( - ( xy*xy ) / (2*w*w) )
    if norm:
        h = h / (np.sqrt(two*np.pi)*w)
    h -= cutoff
    h[np.less(h, zero)] = zero
    return (h)

# creates a simple 2d disturbance (see figure 1)
def planegauss(shape, wx, wy, cx=0, cy=0, theta = 0, cutoff = 0.05, norm = False):
    h1 = lingauss(shape, wx, cx=cx, cy=cy, theta = theta, cutoff=cutoff, norm=norm)
    h2 = lingauss(shape, wy, cx=cx, cy=cy, theta = theta + np.pi/2, cutoff=cutoff, norm=norm)
    return h1*h2

# creates a "seismic" distrubance, with negative and positive height deviation (see figure 2)
def seismic(shape, width, length, cx=0, cy=0, theta=0, a1 = 1, a2 = 1, cutoff=0.05, norm=False):
    """returns simple seismic initial condition on array with shape 'shape'
        theta - angle of rotation
        length - length across distrubance
        width - width across disturbance
        a1 - amplitude of positive portion of distrubance
        a2 - amplitude of negative portion of disturbance
        cx - the x position of the distrubance
        cy - the y position of the disturbance
        cutoff - the magnitude below which values are rounded to zero"""
    offx = width*np.cos(theta)*0.5
    offy = width*np.sin(theta)*0.5
    h1 = a1*planegauss(shape, width, length, cx=cx+offx, cy=cy+offy, theta = theta, cutoff=cutoff, norm=norm) # 'hill'
    h2 = -a2*planegauss(shape, width, length, cx=cx-offx, cy=cy-offy, theta = theta, cutoff=cutoff, norm=norm) # 'valley'
    return h1+h2


# physics constants
class p():
    g = np.float32(9.81) # gravity


# functions to handle coast and boundaries

def land(h, n, u, v, coastx): # how to handle land/above water area
    (u[1:])[coastx] = zero
    (u[:-1])[coastx] = zero # set vel. on either side of land to zero, makes reflective
    (v[:,1:])[coastx] = zero
    (v[:,:-1])[coastx] = zero
#     n[coastx] = zero
    return (n, u, v)


def border(n, u, v, margwidth=15, alph=np.array([0.95, 0.95, 0.95, 0.5])):
    """near one = fake exiting ( attenuate off edges)
    1 = reflective"""
    # attenuate off edges to minimize reflections
    n[0:margwidth] *= alph[0]
    u[0:margwidth] *= alph[0]
    v[0:margwidth] *= alph[0]

    n[-1:-margwidth-1:-1] *= alph[1]
    u[-1:-margwidth-1:-1] *= alph[1]
    v[-1:-margwidth-1:-1] *= alph[1]

    n[:,0:margwidth] *= alph[2]
    u[:,0:margwidth] *= alph[2]
    v[:,0:margwidth] *= alph[2]

    n[:,-1:-margwidth-1:-1] *= alph[3]
    u[:,-1:-margwidth-1:-1] *= alph[3]
    v[:,-1:-margwidth-1:-1] *= alph[3]




# numpy style with  matrix notation

def dndt(h, n, u, v, dx, dy, out) :
# def dndt(state):
    """change in n per timestep, by diff. equations"""
#     h, n, u, v, dx, dy = [qp.asnumpy(state.__dict__[k]) for k in ('h', 'n', 'u', 'v', 'dx', 'dy')]
    hx = np.empty(u.shape, dtype=n.dtype) # to be x (u) momentum array
    hy = np.empty(v.shape, dtype=n.dtype)

    depth = h+n
    hx[1:-1] = (depth[1:] + depth[:-1])*p5 # average
    hx[0] = zero # normal flow boundaries/borders
    hx[-1] = zero # the water exiting the water on the edge is n+h

    hy[:,1:-1] = (depth[:,1:] + depth[:,:-1])*p5
    hy[:,0] = zero
    hy[:,-1] = zero

    hx *= u # height/mass->momentum of water column.
    hy *= v
    out[:,:] = d_dx(hx, -dx)+d_dy(hy, -dy)
#     return ( d_dx(hx, -dx)+d_dy(hy, -dy) )
 # change in x vel. (u) per timestep


# ### Numba Style and CPU compiler

# In[15]:


# Single thread scalar function in python
def dndt2a(jx, iy, h, n, u, v, dx, dy) :
    """change in n per timestep, by diff. equations"""
    p5 = np.float32(0.5)
    depth_jm0im0 = h[jx,  iy  ]+n[jx,    iy]
    depth_jp1im0 = h[jx+1,iy]  +n[jx+1,iy]
    depth_jm1im0 = h[jx-1,iy]  +n[jx-1,iy]
    depth_jm0ip1 = h[jx,  iy+1]+n[jx,  iy+1]
    depth_jm0im1 = h[jx,  iy-1]+n[jx,  iy-1]

    hx_jp1 = u[jx+1,iy]*(depth_jm0im0 + depth_jp1im0)*p5
    hx_jm0 = u[jx,  iy]*(depth_jm1im0 + depth_jm0im0)*p5


    hy_ip1 = v[jx,iy+1]*(depth_jm0im0 + depth_jm0ip1)*p5
    hy_im0 = v[jx,iy  ]*(depth_jm0im1 + depth_jm0im0)*p5

    # assume u and v are zero on edge
    dhx = (hx_jp1-hx_jm0)/dx#[jx,iy]
    dhy = (hy_ip1-hy_im0)/dy#[jx,iy]


    return ( -dhx-dhy )
# numba kernel to drive threads in parallel
def dndta_drive_py(h, n, u, v, dx, dy, out):
    for jx64 in nb.prange(1,out.shape[0]-1):
        for iy64 in range(1,out.shape[1]-1):
            jx = np.int32(jx64)
            iy = np.int32(iy64)
            out[jx,iy] = dndt2a_numba(jx, iy, h, n, u, v, dx, dy)
    for iy in range(0,out.shape[1]):
        out[0, iy] = out[1, iy] # reflective boundary condition
        out[-1,iy] = out[-2,iy]
    for jx in range(0,out.shape[0]):
        out[jx, 0] = out[jx, 1]
        out[jx,-1] = out[jx, -2]


# the following matches the numpy syntax e but isn't as good a boundary condition
def dndt2b(jx, iy, h, n, u, v, dx, dy) :

    """change in n per timestep, by diff. equations"""
    p5 = np.float32(0.5)

    depth_jm0im0 = h[jx,  iy  ]+n[jx,    iy]

    if jx==h.shape[0]-1:
        hx_jp1 = np.float32(0)
        hx_jm0 = u[jx,  iy]*( h[jx-1,iy]  +n[jx-1,iy]+ depth_jm0im0)*p5
    else:
        hx_jp1 = u[jx+1,iy]*(depth_jm0im0 + h[jx+1,iy]  +n[jx+1,iy])*p5
        if jx==0:
            hx_jm0 = np.float32(0)
        else:
            hx_jm0 = u[jx,  iy]*( h[jx-1,iy]  +n[jx-1,iy]+ depth_jm0im0)*p5

    if iy ==h.shape[1]-1:
        hy_ip1 = np.float32(0.0)
        hy_im0 = v[jx,iy  ]*(h[jx,  iy-1]+n[jx,  iy-1] + depth_jm0im0)*p5
    else:
        hy_ip1 = v[jx,iy+1]*(depth_jm0im0 +  h[jx,  iy+1]+n[jx,  iy+1])*p5
        if iy == 0:
            hy_im0 = np.float32(0.0)
        else:
            hy_im0 = v[jx,iy  ]*(h[jx,  iy-1]+n[jx,  iy-1] + depth_jm0im0)*p5

    # assume u and v are zero on edge
    dhx = (hx_jp1-hx_jm0)/dx#[jx,iy]
    dhy = (hy_ip1-hy_im0)/dy#[jx,iy]

    return ( -dhx-dhy )

# numba kernel to drive function
def dndtb_drive_py(h, n, u, v, dx, dy, out):
    for jx64 in nb.prange(0,out.shape[0]):
        for iy64 in range(0,out.shape[1]):
            jx = np.int32(jx64)  # remove?
            iy = np.int32(iy64)
            out[jx,iy] = dndt2b_numba(jx, iy, h, n, u, v, dx, dy)

def dndtc_drive_py(h, n, u, v, dx, dy, out):
    for jx64 in nb.prange(0,out.shape[0]):
        jx = np.int32(jx64)  # remove?
        if jx == 0:
            jx +=1
        elif jx == out.shape[0]-1:
            jx -=1
                # positive reflection
        for iy64 in range(0,out.shape[1]):
            iy = np.int32(iy64)
            if iy==0:
                iy +=1
            elif iy == out.shape[1]-1:
                iy -=1

            out[jx64,iy64] = dndt2b_numba(jx, iy, h, n, u, v, dx, dy)

# compile the scalar function to a cuda device function
ndevice_compiler_numba = nb.njit('float32(int32,int32,float32[:,:],float32[:,:],float32[:,:],float32[:,:],float32,float32)',parallel=True,fastmath=True)
dndt2b_numba = ndevice_compiler_numba (dndt2b)
dndt2a_numba = ndevice_compiler_numba (dndt2a) # same as c but slightly faster

dndt_drive_numba = nb.njit(dndtc_drive_py,parallel=True, fastmath=True)


# caculate the rate of change of the x velocities in the system
def dudt(h, n, f, u, v, dx, dy, out, grav=True, cori=True, advx=True, advy=True, attn=True, nu=0, mu=0.3) :
    mu = np.float32(mu)
    g = p.g

    dudt = np.zeros(u.shape, dtype=u.dtype) # x accel array

    if grav:
        dudt[1:-1] = d_dx(n, -dx/g)


    vn = (v[:,1:]+v[:,:-1])*p5 # n shaped v

    # coriolis force
    if cori:

        fn = f#(f[:,1:]+f[:,:-1])*0.5 # n shaped f
        fvn = (fn*vn) # product of coriolis and y vel.
        dudt[1:-1] += (fvn[1:]+fvn[:-1])*p5 # coriolis force


    # advection

    # advection in x direction
    if advx:
        dudx = d_dx(u, dx)
        dudt[1:-1] -= u[1:-1]*(dudx[1:] + dudx[:-1])*p5 # advection

    # advection in y direction
    # possibly average to corners first, then multiply. may be better?
    if advy:
        duy = np.empty(u.shape, dtype=u.dtype)
        dudy = d_dy(u, dy)
        duy[:,1:-1] = ( dudy[:,1:] + dudy[:,:-1] ) * p5
        duy[:,0] = dudy[:,0]
        duy[:,-1] = dudy[:, -1]
        dudt[1:-1] -= (vn[1:]+vn[:-1])*p5*duy[1:-1] # advection


    #attenuation new
    if attn:
        vna = (v[:,1:]+v[:,:-1])*p5
        depth = p5*np.abs((h[:-1]+h[1:]+n[:-1]+n[1:])) + one
        v_u = (vna[1:]+vna[:-1])*p5
        attenu = 1/(depth) * mu * u[1:-1] * np.sqrt(u[1:-1]**2 + v_u**2) # attenuation
        dudt[1:-1] -= attenu

    # viscous term
#     nu = np.float32(1000/dx)

#     ddux = d_dx(dudx, dx)
#     dduy = np.empty(u.shape, dtype=u.dtype)
#     ddudy = d_dy(duy, dy)
#     dduy[:,1:-1] = ( ddudy[:,1:] + ddudy[:,:-1] ) * p5
#     dduy[:,0] = ddudy[:,0]
#     dduy[:,-1] = ddudy[:, -1]
#     dudt[1:-1] -= nu*(ddux+dduy[1:-1])


    dudt[0] = zero
    dudt[-1] = zero # reflective boundaries
    dudt[:,0] = zero
    dudt[:,-1] = zero # reflective boundaries
    out[:,:] = dudt
#     return ( dudt )




def dvdt(h, n, f, u, v, dx, dy, out,         grav=True, cori=True, advx=True, advy=True, attn=True, nu=0, mu=0.3) :
    mu = np.float32(mu)
    g = p.g

    dvdt = np.zeros(v.shape, dtype=v.dtype) # x accel array

    #gravity
    if grav:

        dvdt[:,1:-1] = d_dy(n, -dy/g)


    un = (u[1:]+u[:-1])*p5 # n-shaped u

    # coriolis force
    if cori:

        fun = (f*un) # product of coriolis and x vel.
        dvdt[:,1:-1] += (fun[:,1:]+fun[:,:-1])*0.5 # coriolis force


    # advection

    # advection in y direction
    if advx:
        dvdy = d_dy(v, dy)
        dvdt[:,1:-1] -= v[:,1:-1]*(dvdy[:,1:] + dvdy[:,:-1])*p5 # advection

    # advection in x direction
    if advy:
        dvx = np.empty(v.shape, dtype=v.dtype)
        dvdx = d_dx(v, dx)
        dvx[1:-1] = ( dvdx[1:] + dvdx[:-1] ) * p5
        dvx[0] = dvdx[0]
        dvx[-1] = dvdx[-1]
        dvdt[:,1:-1] -= (un[:,1:]+un[:,:-1])*p5*dvx[:,1:-1] # advection


    # attenuation
    if attn:
        una = (u[1:]+u[:-1]) * p5
        depth = p5*np.abs(h[:,:-1]+h[:,1:]+n[:,:-1]+n[:,1:]) + one
        uv = (una[:,1:]+una[:,:-1])*p5
        dvdt[:,1:-1] -= mu * v[:,1:-1] * np.sqrt(v[:,1:-1]**2 + uv*uv) / depth


    # viscous term
#     nu = np.float32(dy/1000) # nu given as argument

#     ddvy = d_dy(dvdy, dy)
#     ddvx = np.empty(v.shape, dtype=v.dtype)
#     ddvdx = d_dx(dvx, dx)
#     ddvx[1:-1] = ( ddvdx[1:] + ddvdx[:-1] ) * p5
#     ddvx[0] = ddvdx[0]
#     ddvx[-1] = ddvdx[-1]
#     dvdt[:,1:-1] -= nu*(ddvy+ddvx[:,1:-1])

#     dvdt[:,0] += nu*ddvx[:,0]*ddvy[:,0]
#     dvdt[:,-1] += nu*ddvx[:,-1]*ddvy[:,-1]

    dvdt[0] = zero
    dvdt[-1] = zero # reflective boundaries
    dvdt[:,0] = zero
    dvdt[:,-1] = zero # reflective boundaries
    out[:,:] = dvdt
#     return dvdt


# ### python syntax with looping over array - with numba
#    Looping over the array should take much longer than doing array calculations - however, using the numba library and compiling the function to numba, it goes much faster than even the version with array calculations.

# In[18]:


# calculate the rate of change of the x velocity of a single point
def dudt2(jx, iy, h, n, f, u, v, dx, dy,           grav=True, cori=True, advx=True, advy=True, attn=True, nu=0, mu=0) :
    mu = np.float32(mu)
    p5 = np.float32(0.5)
    one = np.float32(1)
    g=np.float32(9.81)

    jxm1= jx-1
    iym1= iy-1
    jxp1= jx+1
    iyp1= iy+1
    jxm0= jx
    iym0= iy

    dudt = 0

    # gravity
    if grav:
        dudt -= g * ( n[jxm0, iym0] - n[jxm1, iym0] ) / dx


    vn_jm1 = (v[jxm1,iym0]+v[jxm1,iyp1])*p5
    vn_jm0 = (v[jxm0,iym0]+v[jxm0,iyp1])*p5

    # coriolis force
    if cori:


        vf_jm1im0 = f[jxm1,0]*vn_jm1  # techically the iy lookup on f is irrelevant
        vf_jm0im0 = f[jxm0,0]*vn_jm0

        dudt +=  (vf_jm0im0 + vf_jm1im0)*p5

    # advection

    # advection in x direction
    if advx:
        dudx_jp1 = (u[jxp1,iym0]-u[jxm0,iym0])/dx
        dudx_jm0 = (u[jxm0,iym0]-u[jxm1,iym0])/dx
        dudt -= u[jxm0,iym0]*(dudx_jp1+dudx_jm0)*p5


    # advection in y direction
    if advy:
        dudy_ip1 = (u[jxm0,iyp1]-u[jxm0,iym0])/dy
        dudy_im0 = (u[jxm0,iym0]-u[jxm0,iym1])/dy

        vu = (vn_jm1+vn_jm0)*p5

        dudt -= vu*(dudy_ip1 + dudy_im0)*p5 # wrong? multiply in other order?


    #attenuation
    if attn:
        h_jm0 = (h[jxm1,iym0]+h[jxm0,iym0])*p5
        n_jm0 = (n[jxm1,iym0]+n[jxm0,iym0])*p5
        depth = abs(h_jm0+n_jm0)+one
    #     if depth == 0: print ('yikes! zero depth!')
        dudt -= mu * u[jx,iy] * mysqrt(u[jx,iy]**2 + vu*vu) / depth


    # viscous term
    #

    return ( dudt )

device_compiler_numba = nb.njit(
    'float32(int32,int32,float32[:,:],float32[:,:],float32[:,:],float32[:,:],float32[:,:],float32,float32,b1,b1,b1,b1,b1,float32,float32)',
    parallel=True,fastmath=True)

dudt2_numba = device_compiler_numba(dudt2)

def dudt_drive_py(h, n, f, u, v, dx, dy, out,                   grav=True, cori=True, advx=True, advy=True, attn=True, nu=0, mu=0):
    for jx in nb.prange(1, u.shape[0]-1):
        out[jx,0]= np.float32(0.0)
        for iy in nb.prange(1, u.shape[1]-1):
            out[jx,iy] = dudt2_numba(jx,iy,h,n,f,u,v,dx,dy,                                      grav, cori, advx, advy, attn,nu,mu)
     # setting the edges to zero may not be needed if we can assure it stays zero
        out[jx,-1]= np.float32(0.0)
    for iy in nb.prange(0,u.shape[1]):
        out[0,iy]=np.float32(0.0)
        out[-1,iy]=np.float32(0.0)


dudt_drive_numba = nb.njit(dudt_drive_py,parallel=True, fastmath=True)



def dvdt2(jx, iy, h, n, f, u, v, dx, dy,           grav=True, cori=True, advx=True, advy=True, attn=True, nu=0, mu=0) :
    mu = np.float32(mu)
    p5 = np.float32(0.5)
    one = np.float32(1)
    g=np.float32(9.81)

    jxm1= jx-1
    iym1= iy-1
    jxp1= jx+1
    iyp1= iy+1
    jxm0= jx
    iym0= iy

    dvdt = 0

    if grav:
        dvdt -= g * ( n[jxm0, iym0] - n[jxm0, iym1] ) / dy


    un_im1 = (u[jxm0,iym1]+u[jxp1,iym1])*p5
    un_im0 = (u[jxm0,iym0]+u[jxp1,iym0])*p5
    uv = (un_im0 + un_im1)*p5

    # coriolis force
    if cori:
        dvdt +=  f[jxm0,0]*uv


    # advection

    ## advection in y direction
    if advx:
        dvdy_ip1 = (v[jxm0,iyp1]-v[jxm0,iym0])/dy
        dvdy_im0 = (v[jxm0,iym0]-v[jxm0,iym1])/dy
        dvdt -= v[jxm0,iym0]*(dvdy_ip1+dvdy_im0)*p5

    ## advection in x direction
    if advy:
        dvdx_jp1 = (v[jxp1,iym0]-v[jxm0,iym0])/dx
        dvdx_jm0 = (v[jxm0,iym0]-v[jxm1,iym0])/dx
        dvdt -= uv*(dvdx_jp1 + dvdx_jm0)*p5 # wrong? multiply in other order?

    # attenuation
    if attn:
        h_im0 = (h[jxm0,iym1]+h[jxm0,iym0])*p5
        n_im0 = (n[jxm0,iym1]+n[jxm0,iym0])*p5
        depth = abs(h_im0+n_im0) + one
    #     if depth == 0: print('yikes! zero depth!')
        dvdt -= mu * v[jxm0,iym0] * mysqrt(v[jxm0,iym0]**2 + uv*uv) / depth

    return ( dvdt )

dvdt2_numba = device_compiler_numba (dvdt2)

def dvdt_drive_py(h, n, f, u, v, dx, dy, out,                   grav=True, cori=True, advx=True, advy=True, attn=True, nu=0, mu=0):
    for jx in nb.prange(1, v.shape[0]-1):
        out[jx,0]= np.float32(0.0)
        for iy in nb.prange(1, v.shape[1]-1):
          out[jx,iy] = dvdt2_numba(jx,iy,h,n,f,u,v,dx,dy,                                   grav, cori, advx, advy, attn,nu,mu)


        # the following can be avoided if we can assure the edges stay zero

        out[jx,-1]= np.float32(0.0)
    for iy in nb.prange(0,v.shape[1]):
        out[0,iy]=np.float32(0.0)
        out[-1,iy]=np.float32(0.0)

dvdt_drive_numba = nb.njit(dvdt_drive_py,parallel=True, fastmath=True)





def forward(h, n, u, v, f, dt, dx, dy, du, dv, dn,             beta=0, eps=0, gamma=0, mu=0.3, nu=0,             dudt_x=dudt, dvdt_x=dvdt, dndt_x=dndt,             grav=True, cori=True, advx=True, advy=True, attn=True): # forward euler and forward/backward timestep
    """
        beta = 0 forward euler timestep
        beta = 1 forward-backward timestep
    """
    beta = np.float32(beta)
    mu = np.float32(mu)

    du1, du0 = du[:2]
    dv1, dv0 = dv[:2]
    dn0 = dn[0]

    dndt_x(h, n, u, v, dx, dy, dn0) # calculate dndt and put it into dn0

    n1 = n + ( dn0 )*dt

    dudt_x(h, n,  f, u, v, dx, dy, du0,           grav=grav, cori=cori, advx=advx, advy=advy, attn=attn,nu=nu,mu=mu)
    dvdt_x(h, n,  f, u, v, dx, dy, dv0,           grav=grav, cori=cori, advx=advx, advy=advy, attn=attn,nu=nu,mu=mu)
    dudt_x(h, n1, f, u, v, dx, dy, du1,            grav=grav, cori=cori, advx=advx, advy=advy, attn=attn,nu=nu,mu=mu)
    dvdt_x(h, n1, f, u, v, dx, dy, dv1,
           grav=grav, cori=cori, advx=advx, advy=advy, attn=attn,nu=nu,mu=mu)

    u1 = u + ( beta*du1 + (one-beta)*du0 )*dt
    v1 = v + ( beta*dv1 + (one-beta)*dv0 )*dt

    n, u, v = n1, u1, v1

    du = [du1, du0, du0, du0]
    dv = [dv1, dv0, dv0, dv0]
    dn = [dn0, dn0, dn0]
    return n1, u1, v1, du, dv, dn


# In[22]:


def fbfeedback(h, n, u, v, f, dt, dx, dy, du, dv, dn,                beta=1/3, eps=2/3, gamma=0, mu=0.3, nu=0,                dudt_x=dudt, dvdt_x=dvdt, dndt_x=dndt,                grav=True, cori=True, advx=True, advy=True, attn=True):
    """
        predictor (forward-backward) corrector timestep
    """
    beta = np.float32(beta)
    eps = np.float32(eps)
    mu = np.float32(mu)

    du0, du1, du1g = du[:3]
    dv0, dv1, dv1g = dv[:3]
    dn0, dn1 = dn[:2]

    #predict
    n1g, u1g, v1g, dug, dvg, dng = forward(h, n, u, v, f, dt, dx, dy, du, dv, dn, beta, mu=mu, nu=nu,                                           dudt_x=dudt_x, dvdt_x=dvdt_x, dndt_x=dndt_x,                                            grav=grav, cori=cori, advx=advx, advy=advy, attn=attn) # forward-backward first guess

    #feedback on prediction

    dndt_x(h, n1g,u1g,v1g,dx, dy, dn1)
    dn0 = dng[0]
#     dndt_x(h, n,  u,  v,  dx, dy, dn0)

    n1 = n + p5*(dn1 + dn0)*dt

    du0 = dug[1]
    dv0 = dvg[1]
#     dudt_x(h, n,  f, u,  v,  dx, dy, du0,  grav=grav, cori=cori, advx=advx, advy=advy, attn=attn)
#     dvdt_x(h, n,  f, u,  v,  dx, dy, dv0,  grav=grav, cori=cori, advx=advx, advy=advy, attn=attn)
    dudt_x(h, n1g,f, u1g,v1g,dx, dy, du1g, grav=grav, cori=cori, advx=advx, advy=advy, attn=attn,nu=nu,mu=mu)
    dvdt_x(h, n1g,f, u1g,v1g,dx, dy, dv1g, grav=grav, cori=cori, advx=advx, advy=advy, attn=attn,nu=nu,mu=mu)
    dudt_x(h, n1, f, u,  v,  dx, dy, du1,  grav=grav, cori=cori, advx=advx, advy=advy, attn=attn,nu=nu,mu=mu)
    dvdt_x(h, n1, f, u,  v,  dx, dy, dv1,  grav=grav, cori=cori, advx=advx, advy=advy, attn=attn,nu=nu,mu=mu)

    u1 = u + p5*(eps*du1+(one-eps)*du1g+du0)*dt
    v1 = v + p5*(eps*dv1+(one-eps)*dv1g+dv0)*dt

#     n[:,:], u[:,:], v[:,:] = n1, u1, v1
    du, dv, dn = [du1, du0, du0, du0], [dv1, dv0, dv0, dv0], [dn0, dn0, dn0]
    return n1, u1, v1, du, dv, dn


# In[23]:


p5=np.float32(0.5)
p32 =np.float32(1.5)
def genfb(h, n, u, v, f, dt, dx, dy,           du,dv,dn,          beta=0.281105, eps=0.013, gamma=0.0880, mu=0.3, nu=0.3,           dudt_x=dudt, dvdt_x=dvdt, dndt_x=dndt,           grav=True, cori=True, advx=True, advy=True, attn=True): # generalized forward backward feedback timestep
    """
        generalized forward backward predictor corrector
    """

    beta = np.float32(beta)
    eps = np.float32(eps)
    gamma = np.float32(gamma)
    mu = np.float32(mu)
    nu = np.float32(nu)

    dn_m1,dn_m2,dn_m0 = dn     # unpack
    dndt_x(h, n, u, v, dx, dy, dn_m0)

    # must do the following before the u and v !
    n1 = n + ((p32+beta)* dn_m0 - (p5+beta+beta)* dn_m1+ (beta)* dn_m2)*dt

    du_m0,du_m1,du_m2,du_p1 = du     # unpack
    dudt_x(h, n1, f, u, v, dx, dy, du_p1,            grav=grav, cori=cori, advx=advx, advy=advy, attn=attn,nu=nu,mu=mu)

    dv_m0,dv_m1,dv_m2,dv_p1 = dv     # unpack
    dvdt_x(h, n1, f, u, v, dx, dy, dv_p1,            grav=grav, cori=cori, advx=advx, advy=advy, attn=attn,nu=nu,mu=mu)

    u1 = u+ ((p5+gamma+eps+eps)*dt)*du_p1 +((p5-gamma-gamma-eps-eps-eps)*dt)*du_m0              +(gamma*dt)*du_m1 +(eps*dt)*du_m2
    v1 = v+ ((p5+gamma+eps+eps)*dt)*dv_p1 +((p5-gamma-gamma-eps-eps-eps)*dt)*dv_m0              +(gamma*dt)*dv_m1 +(eps*dt)*dv_m2

    #v1 = v+ ((p5+gamma+eps+eps)*dv_p1 +(p5-gamma-gamma-eps-eps-eps)*dv_m0 +\
           #  gamma*dv_m1 +eps*dv_m2)*dt


    dv = ( dv_p1,dv_m0,dv_m1,dv_m2 )
    du = ( du_p1,du_m0,du_m1,du_m2 )
    dn = ( dn_m0,dn_m1,dn_m2 )

    return n1, u1, v1, du,dv,dn


# In[24]:
#
#
# def lin_comb4_thread(v1, v2, v3, v4, w1, w2, w3, w4, out):
#     iy ,jx= cuda.grid(2)
#     if iy<out.shape[1] and jx<out.shape[0]:
#         out[jx,iy] = w1*v1[jx,iy] + w2*v2[jx,iy] + w3*v3[jx,iy] + w4*v4[jx,iy]
# cudacompilelc4 = nb.cuda.jit('void(float32[:,:],float32[:,:],float32[:,:],float32[:,:],float32,float32,float32,float32,float32[:,:])')
# lincomb4_cuda = cudacompilelc4(lin_comb4_thread)
#
# def lin_comb5_thread(v1, v2, v3, v4, v5, w1, w2, w3, w4, w5, out):
#     iy ,jx= cuda.grid(2)
#     if iy<out.shape[1] and jx<out.shape[0]:
#         out[jx,iy] = w1*v1[jx,iy] + w2*v2[jx,iy] + w3*v3[jx,iy] +        w4*v4[jx,iy] + w5*v5[jx,iy]
# cudacompilelc5 = nb.cuda.jit('void(float32[:,:],float32[:,:],float32[:,:],float32[:,:],float32[:,:],float32,float32,float32,float32,float32,float32[:,:])')
# lincomb5_cuda = cudacompilelc5(lin_comb5_thread)

# def lin_comb_master_thread(vs, ws, out):
#     i,j = nb.cuda.grid(2)
#     tmp[i,j] = 0
#     for n, v in enumerate(vs):
#         tmp[i,j] += v[i,j]+ws[n]
#     out[i,j] = tmp[i,j]
# cudacompilelc = nb.cuda.jit('void(float32[:],float32[:,:,:],float32[:,:])')
# lincombmaster_cuda = cudacompilelc(lin_comb_master_thread)

# def lin_comb4(v1, v2, v3, v4, w1, w2, w3, w4, out):
#     threadblock = (32,8)
#     gridx = (out.shape[1]+threadblock[1]-1)//threadblock[1]
#     gridy = (out.shape[0]+threadblock[0]-1)//threadblock[0]
#     lincomb4_cuda[(gridx,gridy),(threadblock)](v1, v2, v3, v4, w1, w2, w3, w4, out)
#
#
# def lin_comb5(v1, v2, v3, v4, v5, w1, w2, w3, w4, w5, out):
#     threadblock = (32,8)
#     gridx = (out.shape[1]+threadblock[1]-1)//threadblock[1]
#     gridy = (out.shape[0]+threadblock[0]-1)//threadblock[0]
#     lincomb5_cuda[(gridx,gridy),(threadblock)](v1, v2, v3, v4, v5, w1, w2, w3, w4, w5, out)

def donothing (h, n, u, v, f, dt, dx, dy, nu, coastx, bounds, mu, itr): return


def simulate(initstate, t, timestep=forward, drive=donothing,              bounds = [0.97, 0.97, 0.97, 0.97], saveinterval=10,             beta=0.281105, eps=0.013, gamma=0.0880, mu=0.3, nu=0.3,              dudt_x = dudt, dvdt_x = dvdt, dndt_x = dndt,              grav=True, cori=True, advx=True, advy=True, attn=True): # gives surface height array of the system after evert dt
    """
        evolve shallow water system from initstate over t seconds
        returns:
            ntt (numpy memmap of n through time) numpy array,
            maxn (the maximum value of n over the duration at each point) numpy array,
            minn (the minimum value of n over the duration at each point) numpy array,
            timemax (the number of seconds until the maximum height at each point) numpy array
    """
    bounds = np.asarray(bounds, dtype=np.float32)
    h, n, u, v, f, dx, dy, dt = [initstate[k] for k in ('h', 'n', 'u', 'v', 'lat', 'dx', 'dy', 'dt')]

    f = np.float32(((2*2*np.pi*np.sin(f*np.pi/180))/(24*3600))[:,np.newaxis])


    du0 = np.zeros_like(u)
    dv0 = np.zeros_like(v)
    dn0 = np.zeros_like(n)


    dndt_x(h, n, u, v, dx, dy, dn0)
    dn = (dn0, np.copy(dn0), np.copy(dn0))

    dudt_x(h, n, f, u, v, dx, dy, du0)
    du = (du0, np.copy(du0), np.copy(du0), np.copy(du0))

    dvdt_x(h, n, f, u, v, dx, dy, dv0)
    dv = (dv0, np.copy(dv0), np.copy(dv0), np.copy(dv0))

    nu = (dx+dy)/1000

    mmax = np.max(np.abs(n))
    landthresh = 1.5*np.max(n) # threshhold for when sea ends and land begins
    itrs = int(np.ceil(t/dt))
    saveinterval = np.int(saveinterval//dt)
    assert (dt >= 0), 'negative dt!' # dont try if timstep is zero or negative

    ntt = np.zeros((np.int(np.ceil(itrs/saveinterval)),)+n.shape, dtype=np.float32)
    maxn = np.zeros(n.shape, dtype=n.dtype) # max height in that area

    coastx = np.less(h, landthresh) # where the reflective condition is enforced on the coast

    print('simulating...')
    try:
        for itr in range(itrs):# iterate for the given number of iterations
            if itr%saveinterval == 0:
                ntt[np.int(itr/saveinterval),:,:] = n

            maxn = np.max((n, maxn), axis=0) # record new maxes if they are greater than previous records

            # pushes n, u, v one step into the future
            n,u,v, du, dv, dn = timestep(h, n, u, v, f, dt, dx, dy, du, dv, dn,                         0.281105, 0.013, 0.0880, 0.3, 0.3,                         dudt_x, dvdt_x, dndt_x,                         grav=True, cori=True, advx=True, advy=True, attn=True                    #      beta=beta, eps=eps, gamma=gamma, mu=mu, nu=nu, \
                   #     dudt_x=dudt_x, dvdt_x=dvdt_x, dndt_x=dndt_x, \
                  #      grav=grav, cori=cori, advx=advx, advy=advy, attn=attn
                                        )

            land(h, n, u, v, coastx) # how to handle land/coast
            border(n, u, v, 15, bounds)
     #       drive(h, n, u, v, f, dt, dx, dy, nu, coastx, bounds, mu, itr)
        print('simulation complete')
    except Exception as e:
        print('timestep: ', itr)
        raise e
    return ntt, maxn#, minn, timemax # return surface height through time and maximum heights


#wavespeed and differential tests
import unittest
fooo = []
class testWaveSpeed(unittest.TestCase): # tests if the wave speed is correct
    def setUp(self):
        self.dur = 100 # duration of period to calculate speed over
        self.size = (10, 1000) # grid squares (dx's)
        self.dx = np.float32(100) # meters
        self.dy = np.float32(100)
        self.lat = np.linspace(0, 0, self.size[0]) # physical location the simulation is over
        self.lon = np.linspace(0, 0 , self.size[1])
        self.h = np.float32(10000*np.ones(self.size))
        self.n = np.float32(0.1)*lingauss(self.size, 10, cy=500, theta=np.pi/2) # intial condition single wave in the center
        self.u = np.zeros((self.size[0]+1, self.size[1]+0)) # x vel array
        self.v = np.zeros((self.size[0]+0, self.size[1]+1)) # y vel array
        self.dt = 0.3*self.dx/np.sqrt(np.max(self.h)*p.g)
        self.margin = 0.15 # error margin of test

        self.initialcondition = {
            'h':self.h,
            'n':self.n,
            'u':self.u,
            'v':self.v,
            'dt':self.dt,
            'dx':self.dx,
            'dy':self.dy,
            'lat':self.lat,
            'lon':self.lon
        }
#         self.testStart = State(self.h, self.n, self.u, self.v, self.dx, self.dy, self.lat, self.lon)
    def calcWaveSpeed(self, ar1, ar2, Dt): # calculat how fast the wave is propagating out
        midstrip1 = ar1[int(ar1.shape[0]/2),int(ar1.shape[1]/2):]
        midstrip2 = ar2[int(ar1.shape[0]/2),int(ar2.shape[1]/2):]
        peakloc1 = np.argmax(midstrip1)
        peakloc2 = np.argmax(midstrip2)
        plt.figure(6)
        plt.clf()
#         plt.subplot(2, 1, 1)
#         plt.imshow(ar1)
#         plt.subplot(2, 1, 2)
#         plt.imshow(ar2)
        plt.plot(midstrip1)
        plt.plot(midstrip2, "--")
#         plt.plot(midstrip1-midstrip2)
        plt.show()
        speed = (peakloc2 - peakloc1)*self.dy/Dt
        return speed
    def calcExactWaveSpeed(self): # approximently how fast the wave should be propagating outwards
        ws = np.sqrt(9.81*np.average(self.h))
        return ws
    def test_wavespeed(self): # test if the expected and calculated wave speeds line up approcimently

        self.simdata = simulate(self.initialcondition, self.dur, saveinterval=0.2,                                 timestep=genfb, bounds=np.array([1, 1, 1, 1]), mu=0, cori=False, advx=False, advy=False, attn=False)
#         self.testFrames, self.testmax, self.testmin = self.simdata[:3]
        fig = plt.figure(7)
        plt.imshow(self.simdata[0][:,5])#self.testStart.n)
#         arts = [(plt.imshow(frame),) for frame in self.simdata[0]]
#         anim = animation.ArtistAnimation(fig, arts)

        self.testFrames = self.simdata[0]
        self.testEndN = self.testFrames[-1]
        calcedws = self.calcWaveSpeed( self.initialcondition['n'], self.testEndN, self.dur )
        exactws = self.calcExactWaveSpeed()
        err = (calcedws - exactws)/exactws
        print(calcedws, exactws)
        print(err, self.margin)

        assert (abs(err) < self.margin) # error margin
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
def test():
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
#You can pass further arguments in the argv list, e.g.
#unittest.main(argv=['ignored', '-v'], exit=False)
#unittest.main()

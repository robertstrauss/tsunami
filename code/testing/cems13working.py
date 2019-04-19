
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('matplotlib', 'notebook')
import numpy as np
xp = np

import time
import matplotlib as mpl
import numba as nb

import math
sqrt = math.sqrt


argfast = {'parallel':True, 'fastmath':True, 'nopython':True}
argfast_noparallel = {'parallel':False, 'fastmath':True, 'nopython':True}


# In[2]:

class p():
    g = np.float32(9.8)
    mu = np.float32(1.0)
    nu = np.float(0.1)

#convenient floats
zero = np.float32(0)
p5 = np.float32(0.5)
one = np.float32(1)
two = np.float32(2)



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




# useful math functions

def d_dx_py(a, dx):
    ddx = ( a[:-1] - a[1:] )*(np.float32(-1)/dx) 
    return ddx
d_dx = nb.jit(d_dx_py,**argfast)
    

def d_dy_py(a, dy):
    ddy = ( a[:,:-1] - a[:,1:] )*(np.float32(-1)/dy)
    return ddy
d_dy = nb.jit(d_dy_py,**argfast)    
    

def div_py(u, v, dx, dy):  
    div = d_dx(u, dx) + d_dy(v, dy)
    return div
div = nb.jit(div_py,**argfast)    
    
mu = np.float32(p.mu)
g = np.float32(p.g)
nu =np.float32(p.nu)
@nb.jit(**argfast_noparallel)
def dudt(h, n, f, u, v, dx, dy, mu=mu,nu=nu,g=g) : 
    mu = np.float32(mu)
    p5=np.float32(0.5)
    g = np.float32(g)
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
    
    
    #attenuation
    una = (u[1:]+u[:-1]) * p5
    vna = (v[:,1:]+v[:,:-1])*p5

    attenu = np.float32(1)/(h+n) * mu * una * np.sqrt(una*una + vna*vna) # attenuation
    dudt[1:-1] -= (attenu[1:] + attenu[:-1])*p5
    

    
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
    

def dudt2_advection_py(i,j,h, n, f, u, v, dx, dy) : 
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
dudt2_advection = nb.jit(dudt2_advection_py,**argfast)   # never use parallel if there is a broadcast-- numba bug


@nb.jit(**argfast)   # never use parallel if there is a broadcast-- numba bug
def dvdt2_advection(i,j,h, n, f, u, v, dx, dy)  :
    return dudt2_advection(j,i,h.T, n.T, f.T, v.T, u.T, dy, dx ) 

 
def dvdt2_advection_py(i,j,h, n, f, u, v, dx, dy) : 
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
dvdt2_advection = nb.jit(dvdt2_advection_py,**argfast)  # never use parallel if there is a broadcast-- numba bug

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
    
g = np.float32(p.g)


def dudt2_grav_py(i,j,h, n, f, u, v, dx, dy):
# physics constants
  #  g = np.float32(9.81) # gravity
    # gravity  (is this wrong in main routine== check indexing)
    # malfunction if i=0 or i=n 
    grav = (n[i-1,j]-n[i,j])*(g)/dx  #n[i-1] and n[i] straddle u[i]
    return grav
dudt2_grav= nb.jit(dudt2_grav_py,**argfast)   # never use parallel if there is a broadcast-- numba bug


@nb.jit(**argfast)   # never use parallel if there is a broadcast-- numba bug
def dvdt2_grav(i,j,h, n, f, u, v, dx, dy):
    return dudt2_grav(j,i,h.T, n.T, f.T, v.T, u.T, dy, dx )

def dvdt2_grav_py(i,j,h, n, f, u, v, dx, dy):
# physics constants
   # g = np.float32(9.81) # gravity
    # gravity  (is this wrong in main routine== check indexing)
    # malfunction if i=0 or i=n 
    grav = (n[i,j-1]-n[i,j])*(g)/dy  #n[i-1] and n[i] straddle u[i]
    return grav
dvdt2_grav= nb.jit(dvdt2_grav_py,**argfast) 

#@nb.njit()   # never use parallel if there is a broadcast-- numba bug
def dudt_coriolis_py(h, n, f, u, v, dx, dy, out,righthand=True,vn=None) : 
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
dudt_coriolis= nb.jit(dudt_coriolis_py,**argfast_noparallel) 

#@nb.njit()   # never use parallel if there is a broadcast-- numba bug
@nb.jit(**argfast_noparallel)
def dvdt_coriolis(h, n, f, u, v, dx, dy, out,vn=None):
    if vn is not None:  vn = vn.T
    dudt_coriolis(h.T, n.T, f.T, v.T, u.T, dy, dx,  out.T,righthand=False,vn=vn)   
    
    
def dudt2_coriolis_py(i,j,h, n, f, u, v, dx, dy) :
    p5 = np.float32(0.5)
    # first average v along j axis to make it n-centered
    #v_n[i,j] = (v[i,j]+v[i,j+1])*p5  # average neighbors. #### v_n is n-centered between v'000000s
    last_v_n_i = (v[i-1,j]+v[i-1,j+1])*p5
    v_n = (v[i,j]+v[i,j+1])*p5  # stradles n[i,j]

    coriolis_u = (f[i-1,j]*last_v_n_i+f[i,j]*v_n)*p5 # coriolis force F is n-centered
    return coriolis_u
dudt2_coriolis= nb.jit(dudt2_coriolis_py,**argfast_noparallel) 
def dudt2_dummy_py(i,j,h, n, f, u, v, dx, dy) :
    p5 = np.float32(0.5)
    # first average v along j axis to make it n-centered
    #v_n[i,j] = (v[i,j]+v[i,j+1])*p5  # average neighbors. #### v_n is n-centered between v'000000s
    last_v_n_i = (v[i,j]+v[i,4])*p5
    v_n = (v[i,j]+v[i,5])*p5  # stradles n[i,j]

  #  coriolis_u = (f[i-1,j]*last_v_n_i+f[i,j]*v_n)*p5 # coriolis force F is n-centered
    return (f[i-1,j]*last_v_n_i+f[i,j]*v_n)*p5

@nb.jit(**argfast_noparallel)   # never use parallel if there is a broadcast-- numba bug
def dvdt2_coriolis(i,j,h, n, f, u, v, dx, dy):
    return -dudt2_coriolis(j,i,h.T, n.T, f.T, v.T, u.T, dy, dx)

def dvdt2_coriolis_py(i,j,h, n, f, u, v, dx, dy) :
    p5 = np.float32(0.5)
    # first average v along j axis to make it n-centered
    #v_n[i,j] = (v[i,j]+v[i,j+1])*p5  # average neighbors. #### v_n is n-centered between v'000000s
    last_u_n_i = (u[i,j-1]+u[i+1,j-1])*p5
    u_n = (u[i,j]+u[i+1,j])*p5  # stradles n[i,j]

    coriolis_v = (f[i,j-1]*last_u_n_i+f[i,j]*u_n)*p5 # coriolis force F is n-centered
    return coriolis_v
dvdt2_coriolis= nb.jit(dvdt2_coriolis_py,**argfast_noparallel) 


mu = np.float32(p.mu)
def dudt2_drag_py(i,j,h, n, f, u, v, dx, dy) : 
    p5 = np.float32(0.5)
# physics constants
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
    drag = mu* uu*sqrt(uu*uu+ v_u*v_u)/(ro*depth_u)
    return -drag


    
    
dudt2_drag= nb.jit(dudt2_drag_py,**argfast)     
    
@nb.jit(**argfast)   # never use parallel if there is a broadcast-- numba bug
def dvdt2_drag(i,j,h, n, f, u, v, dx, dy):
    return dudt2_drag(j,i,h.T, n.T, f.T, v.T, u.T, dy, dx )

 
def dvdt2_drag_py(i,j,h, n, f, u, v, dx, dy) : 
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
    drag = mu* vv*sqrt(vv*vv+ u_v*u_v)/(ro*depth_v)
    return -drag
dvdt2_drag= nb.jit(dvdt2_drag_py,**argfast)    


@nb.jit(**argfast)   # never use parallel if there is a broadcast-- numba bug
def dudt_mydrag(h, n, f, u, v, dx, dy,out): 
    p5=np.float32(0.5)
  #  mu = np.float32(0.3)
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
    dudt_advection(h=h, n=n, f=f, u=u, v=v, dx=dx, dy=dy,out=out,vn=vn)
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
  # never use parallel if there is a broadcast-- numba bug       
def dudt2_almostall_py(i,j,h, n, f, u, v, dx, dy)  : 

    dudt=dudt2_drag(i,j,h, n, f, u, v, dx, dy)+        dudt2_advection(i,j,h, n, f, u, v, dx, dy)+        dudt2_grav(i,j,h, n, f, u, v, dx, dy)
    return dudt
dudt2_almostall=nb.jit(dudt2_almostall_py,**argfast)

  # never use parallel if there is a broadcast-- numba bug       
def dvdt2_almostall_py(i,j,h, n, f, u, v, dx, dy)  : 
    dvdt=dvdt2_drag(i,j,h, n, f, u, v, dx, dy)+        dvdt2_advection(i,j,h, n, f, u, v, dx, dy)+        dvdt2_grav(i,j,h, n, f, u, v, dx, dy)
    return dvdt
dvdt2_almostall=nb.jit(dvdt2_almostall_py,**argfast)


def dudt2_no_n(i,j,h, n, f, u, v, dx, dy)  : 
    dudt=dudt2_coriolis(i,j,h, n, f, u, v, dx, dy)+        dudt2_advection(i,j,h, n, f, u, v, dx, dy)
    return dudt

   # never use parallel if there is a broadcast-- numba bug       
def dudt2_all_py(i,j,h, n, f, u, v, dx, dy)  : 
    dudt=dudt2_almostall(i,j,h, n, f, u, v, dx, dy)+          dudt2_coriolis(i,j,h, n, f, u, v, dx, dy)
    return dudt
dudt2_all=nb.jit(dudt2_all_py,**argfast)

   # never use parallel if there is a broadcast-- numba bug
def dvdt2_all_py(i,j,h, n, f, u, v, dx, dy)  : 
    return dudt2_almostall(j,i,h.T, n.T, f.T,  v.T, u.T,dy, dx)-            dudt2_coriolis(j,i,h.T, n.T, f.T,  v.T, u.T,dy, dx)  # check sign
dvdt2_all=nb.jit(dvdt2_all_py,**argfast)



  # never use parallel if there is a broadcast-- numba bug       
def dvdt2_all_py(i,j,h, n, f, u, v, dx, dy)  : 
    dvdt=dvdt2_almostall(i,j,h, n, f, u, v, dx, dy)
    t = dvdt2_coriolis(i,j,h, n, f, u, v, dx, dy)
    return dvdt-t
dvdt2_all=nb.jit(dvdt2_all_py,**argfast)

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


# I forgot what this one is for.
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



TPB_slow=16   # cuda_y(slow)  not our y 
TPB_fast=32 # 512//TPBY   # cuda_x (fast) not our x
threadsperblock = (TPB_fast,TPB_slow) # (16,16)  # backwards to Cmatrix order!  needs to be fast.slow or x,y
Sdim = threadsperblock
#  dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);     ( block_length_in_fast_direction , block_length_in_slow_direction) 

from numba import cuda

compiler = cuda.jit('void(float32[:,:], float32[:,:], float32[:,:],float32[:,:],float32[:,:],float32,float32,float32[:,:])' ,inline=True,fastmath=True)        
def compileit (fn,blockdim=None,border=(0,0),innerloop=False):
    name = fn.__name__
    fc = compiler(fn)
    fc.__name__ = name+'cuda'
    if blockdim == None:
        blockdim = (32,fc._func.get().attrs.maxthreads//32)
        print ("autoset",fc._func.get().attrs.maxthreads, "treads to block",blockdim)
    fc.BLOCKDIM = blockdim
    fc.BORDER = border
    fc.INNERLOOP= innerloop
    print ("compiled ",fc.__name__,'blockdim',fc.BLOCKDIM,'border',fc.BORDER,"innerloop",fc.INNERLOOP)
    return fc
    
def inner(ii,jj,i,j,h, n, u, v, f, dx, dy, out_u):

        p5 = np.float32(0.5)
   
        um0m0 = u[i,j]
        um1m0 = u[i-1,j]
        um1p1 = u[i-1,j+1]
        up1m0 = u[i+1,j]   # this one is a pitfall if i = 32
        um0m1 = u[i,j-1]
        um0p1 = u[i,j+1]
        
        vm0m0 = v[i,j]
        vm1m0 =v[i-1,j]
        vm1p1 = v[i-1,j+1]
        vm0p1 = v[i,j+1]
    
        # uu
        nm0m0 = n[ii,jj]
        nm1m0 = n[ii-1,jj]
        hm0m0 = h[ii,jj]
        hm1m0 = h[ii-1,jj]
        fm0m0 = f[ii,jj]
        fm1m0 = f[ii-1,jj]
            
          
        #if  i>0  and j>0  and i< by-1 and j<bx-1 :   #  the edges better be valid!!!  or problem here.   xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        last_v_n_i = (vm1m0+vm1p1)*p5
        v_n = (vm0m0+vm0p1)*p5  # stradles n[i,j]
            # advection
        # malfunctions when i=0 or i=n
        dudx_last = (um0m0-um1m0)/dx   # n-centered dx[i-1] is straddled by u[i,j]-u[i-1,j])
        dudx_next = (up1m0-um0m0)/dx
        u_du_dx = um0m0*(dudx_last+dudx_next)*p5 # back to u-centered

        # malfunctions when j=0 or j=m
        dudy_last = (um0m0-um0m1)/dy
        dudy_next = (um0p1-um0m0)/dy
        v_du_dy= (dudy_last+dudy_next)*p5*(last_v_n_i+v_n)*p5 # yes p5 twice

        advection = -(u_du_dx+v_du_dy)


    # physics constants
     #   g = np.float32(9.81) # gravity
        # gravity  (is this wrong in main routine== check indexing)
        # malfunction if i=0 or i=n 
        grav = (nm1m0-nm0m0)*(g)/dx  #n[i-1] and n[i] straddle u[i]

        # first average v along j axis to make it n-centered
        #v_n[i,j] = (v[i,j]+v[i,j+1])*p5  # average neighbors. #### v_n is n-centered between v'000000s
        last_v_n_i = (vm1m0+vm1p1)*p5
        v_n = (vm0m0+vm0p1)*p5  # stradles n[i,j]

        coriolis_u = (fm1m0*last_v_n_i+fm0m0*v_n)*p5 # coriolis force F is n-centered


    # physics constants
    #    mu = np.float32(0.3)
    #attenuation by friction
        last_v_n_i = (vm1m0+vm1p1)*p5
        v_n = (vm0m0+vm0p1)*p5  # stradles n[i,j]
        # this calculation is different that the one used before because it
        # is u-centered directly rather then averaging it.
        # malfuncitons if i=0 or i = n
        last_depth_x = nm1m0+hm1m0  
        depth = nm0m0+hm0m0
        depth_u = (depth+last_depth_x)*p5  # straddles u[i]
        v_u =   (last_v_n_i +v_n)*p5  # straddles u[i]
        u1 = u[i,j]
        ro = np.float32(1)  # remove
        drag = -mu*u1*sqrt(u1*u1+ v_u*v_u)/(ro*depth_u)

        k = drag +advection+grav+coriolis_u #
        out_u[ii,jj]= k


#@cuda.jit('void(float32[:,:], float32[:,:], float32[:,:],float32[:,:],float32[:,:],float32,float32,float32[:,:])')
def cu_u_driver_global_py(h, n, u, v, f, dx, dy, out_u):
    """This kernel function will be executed by a thread."""
       # THE LOGIC OF i and j IS REVERSED IN THIS VERSION.
     #ii,jj  = cuda.grid(2)   #slow,fast    # in our system i is slow/far and j is fast/near,  usually defined differently
                                          # cuda  y is slow and cuda  x is fast  (this is also flipped from our system)
                                          #  in our system    i(x) --> cuda.y  and j(y) --> cuda.x
                                          #  i is axis[0] and j is axis[1] in our system.  flipped from normal
    j,i  = cuda.grid(2)  # cuda.x cuda.y  fast slow
    if (i < out_u.shape[0]-1) and (j < out_u.shape[1]-1):
        if  i>0  and j>0 : 
            p5 = np.float32(0.5)
            last_v_n_i = (v[i-1,j]+v[i-1,j+1])*p5
            v_n = (v[i,j]+v[i,j+1])*p5  # stradles n[i,j]
#                advection
#            malfunctions when i=0 or i=n
            dudx_last = (u[i,j]-u[i-1,j])/dx   # n-centered dx[i-1] is straddled by u[i,j]-u[i-1,j])
            dudx_next = (u[i+1,j]-u[i,j])/dx
            u_du_dx = u[i,j]*(dudx_last+dudx_next)*p5 # back to u-centered

#            malfunctions when j=0 or j=m
            dudy_last = (u[i,j]-u[i,j-1])/dy
            dudy_next = (u[i,j+1]-u[i,j])/dy
            v_du_dy= (dudy_last+dudy_next)*p5*(last_v_n_i+v_n)*p5 # yes p5 twice

            advection = -(u_du_dx+v_du_dy)


       # physics constants
       #     g = np.float32(9.81) # gravity
            # gravity  (is this wrong in main routine== check indexing)
            # malfunction if i=0 or i=n 
            grav = (n[i-1,j]-n[i,j])*(g)/dx  #n[i-1] and n[i] straddle u[i]

            # first average v along j axis to make it n-centered
            #v_n[i,j] = (v[i,j]+v[i,j+1])*p5  # average neighbors. #### v_n is n-centered between v'000000s
            last_v_n_i = (v[i-1,j]+v[i-1,j+1])*p5
            v_n = (v[i,j]+v[i,j+1])*p5  # stradles n[i,j]

            coriolis_u = (f[i-1,j]*last_v_n_i+f[i,j]*v_n)*p5 # coriolis force F is n-centered


        # physics constants
       #     mu = np.float32(0.3)
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
            drag = -mu*uu*math.sqrt(uu*uu+ v_u*v_u)/(ro*depth_u)

            k = drag +advection+grav+coriolis_u # grav+advection+coriolis_u+drag
        out_u[i,j]= k    
   
cu_u_driver_global = compileit(cu_u_driver_global_py)

#cu_u_driver_global.__name__="cu_u_driver_global"
#cu_u_driver_global.BLOCKDIM  = threadsperblock  # slow fast

def cu_u_driver_global_inner_py(h, n, u, v, f, dx, dy, out_u):
    sj,si  = cuda.grid(2)  # cuda.x cuda.y skow,fast
    gj,gi = sj,si
    if (gi < out_u.shape[0]-1) and (gj < out_u.shape[1]-1):
        if  gi>0  and gj>0 : 
            inner(gi,gj,si,sj,h, n, u, v, f, dx, dy, out_u)

def funcs(si,sj,h, n, f, u, v, dx, dy):
            tmp=np.float32(0.0)
            tmp +=  cuda_dudt2_drag_ret(si,sj,h, n, f, u, v, dx, dy)  
            tmp += cuda_dudt2_advection_ret(si,sj,h, n, f, u, v, dx, dy)  
            tmp += cuda_dudt2_grav_ret(si,sj,h, n, f, u, v, dx, dy) 
            tmp += cuda_dudt2_coriolis_ret(si,sj,h, n, f, u, v, dx, dy)
            return tmp

def cu_u_driver_global_funcs_py(h, n, u, v, f, dx, dy, out_u):
  
    sj,si  = cuda.grid(2)  # cuda.x cuda.y skow,fast
    gj,gi = sj,si
    if (gi < out_u.shape[0]-1) and (gj < out_u.shape[1]-1):
        if  gi>0  and gj>0 : 
            out_u[gi,gj]=funcs(si,sj,h, n, f, u, v, dx, dy)
# 
#             tmp=np.float32(0.0)
#             tmp +=  cuda_dudt2_drag_ret(si,sj,h, n, f, u, v, dx, dy)  
#             tmp += cuda_dudt2_advection_ret(si,sj,h, n, f, u, v, dx, dy)  
#             tmp += cuda_dudt2_grav_ret(si,sj,h, n, f, u, v, dx, dy) 
#             tmp += cuda_dudt2_coriolis_ret(si,sj,h, n, f, u, v, dx, dy)
#           
    #    tmp += cuda_dvdt2_drag_ret(sj,si,h, n, f, u, v, dx, dy)  
    #    tmp += cuda_dvdt2_advection_ret(sj,si,h, n, f, u, v, dx, dy)  
    #    tmp += cuda_dvdt2_grav_ret(sj,si,h, n, f, u, v, dx, dy) 
    #    tmp += cuda_dvdt2_coriolis_ret(sj,si,h, n, f, u, v, dx, dy)  
        # the following is the dominant time step  
#        out_u[gi,gj] = tmp # gi+gj/100.0 #tmp  ##  IS THIS REGISTERED RIGHT??  SHOULD IT be gj+1, gi+1:  I think so because si and sj are restricted.

def cu_u_driver_global_inner_col_py(h, n, u, v, f, dx, dy, out_u):
    sj,si  = cuda.grid(2)  # cuda.x cuda.y skow,fast
    gj,gi = sj,si
    for gi in range(1, out_u.shape[0]-1):    
        if  (gj < out_u.shape[1]-1):
            if  gi>0  and gj>0 : 
                inner(gi,gj,si,sj,h, n, u, v, f, dx, dy, out_u)

def cu_u_driver_global_funcs_col_py(h, n, u, v, f, dx, dy, out_u):
    sj,si  = cuda.grid(2)  # cuda.x cuda.y skow,fast
    gj = sj
    for gi in range(1, out_u.shape[0]-2):    
        if  (gj < out_u.shape[1]-1):
            if  gi>0  and gj>0 : 
                tmp = cuda_dudt2_advection_ret(si,sj,h, n, f, u, v, dx, dy)  
                tmp += cuda_dudt2_grav_ret(si,sj,h, n, f, u, v, dy , dy) 
                tmp +=  cuda_dudt2_drag_ret(si,sj,h, n, f, u, v, dx, dy)  
                tmp += cuda_dvdt2_drag_ret(sj,si,h, n, f, u, v, dx, dy)  
                out_u[gi,gj] = tmp #funcs(si,sj,h, n, u, v, f, dx, dy)


#@cuda.jit('void(float32[:,:], float32[:,:], float32[:,:],float32[:,:],float32[:,:],float32,float32,float32[:,:])')
def cu_u_driver_shared_py(uh, un, uu, uv, uf, dx, dy, out_u):
    """This kernel function will be executed by a thread."""
    ii,jj  = cuda.grid(2)   #slow,fast    # in our system i is slow/far and j is fast/near,  usually defined differently
                                          # cuda  y is slow and cuda  x is fast  (this is also flipped from our system)
                                          #  in our system    i(x) --> cuda.y  and j(y) --> cuda.x
                                          #  i is axis[0] and j is axis[1] in our system.  flipped from normal

    # Allocate static shared memory of 512 (max number of threads per block for CC < 3.0)
    # This limits the maximum block size to 512.
#    sh = cuda.shared.array(shape=(TPB_slow,TPB_fast), dtype=nb.float32)
#    sn = cuda.shared.array(shape=(TPB_slow,TPB_fast), dtype=nb.float32)
    su = cuda.shared.array(shape=(TPB_slow,TPB_fast), dtype=nb.float32) #33
    sv = cuda.shared.array(shape=(TPB_slow,TPB_fast), dtype=nb.float32) # 33
#    sf = cuda.shared.array(shape=(TPB_slow,TPB_fast), dtype=nb.float32)

    i = cuda.threadIdx.x   # fast   in our system    i(x) --> cuda.y  and i(y) --> cuda.x
    j = cuda.threadIdx.y   # slow    in our system    i(x) --> cuda.y  and i(y) --> cuda.x
    bi = cuda.blockDim.y
    bj = cuda.blockDim.x
#    i = tx + bx * bw
    if ii < uu.shape[0] and jj<uu.shape[1]:
        su[i,j] = uu[ii,jj]
#     if ii < un.shape[0] and jj<un.shape[1]:
#         sn[i,j] = un[ii,jj]    
#         sh[i,j] = uh[ii,jj]
#         sf[i,j] = uf[ii,jj]
    if ii < uv.shape[0] and jj<uv.shape[1]:
        sv[i,j] = uv[ii,jj]
    
    n = un #sn
    h = uh #sh
    u = su
    v = sv
    f = uf #sf
    
#     h = uh
#     n = un
#     u = uu
#     v = uv
#     f = uf
#      i,ii = ii,i
#      j,jj = jj,j
#    i = ii
#    j = jj
    cuda.syncthreads() # wait for shared memory fill to complete
   
    if (ii < out_u.shape[0]-1) and (jj < out_u.shape[1]):    ######    MAJOR PROBLEM LOGICALLY since inner edges are not getting done.
        k = np.float32(0.0)
        if  i>0  and j>0  and i< bi-1 and j<bj-1 : 
            p5 = np.float32(0.5)
            vm0m0 = v[i,j]
            vm1m0 =v[i-1,j]
            vm1p1 = v[i-1,j+1]
            vm0p1 = v[i,j+1]
            um0m0 = u[i,j]
            um1m0 = u[i-1,j]
            um1p1 = u[i-1,j+1]
            up1m0 = u[i+1,j]
            um0m1 = u[i,j-1]
            um0p1 = u[i,j+1]
        
            # uu
            nm0m0 = n[ii,jj]
            nm1m0 = n[ii-1,jj]
            hm0m0 = h[ii,jj]
            hm1m0 = h[ii-1,jj]
            fm0m0 = f[ii,jj]
            fm1m0 = f[ii-1,jj]
            
   
            last_v_n_i = (vm1m0+vm1p1)*p5
            v_n = (vm0m0+vm0p1)*p5  # stradles n[i,j]
                # advection
            # malfunctions when i=0 or i=n
            dudx_last = (um0m0-um1m0)/dx   # n-centered dx[i-1] is straddled by u[i,j]-u[i-1,j])
            dudx_next = (up1m0-um0m0)/dx
            u_du_dx = um0m0*(dudx_last+dudx_next)*p5 # back to u-centered

            # malfunctions when j=0 or j=m
            dudy_last = (um0m0-um0m1)/dy
            dudy_next = (um0p1-um0m0)/dy
            v_du_dy= (dudy_last+dudy_next)*p5*(last_v_n_i+v_n)*p5 # yes p5 twice

            advection = -(u_du_dx+v_du_dy)


        # physics constants
        #    g = np.float32(9.81) # gravity
            # gravity  (is this wrong in main routine== check indexing)
            # malfunction if i=0 or i=n 
            grav = (nm1m0-nm0m0)*(g)/dx  #n[i-1] and n[i] straddle u[i]

            # first average v along j axis to make it n-centered
            #v_n[i,j] = (v[i,j]+v[i,j+1])*p5  # average neighbors. #### v_n is n-centered between v'000000s
            last_v_n_i = (vm1m0+vm1p1)*p5
            v_n = (vm0m0+vm0p1)*p5  # stradles n[i,j]

            coriolis_u = (fm1m0*last_v_n_i+fm0m0*v_n)*p5 # coriolis force F is n-centered


        # physics constants
      #      mu = np.float32(0.3)
        #attenuation by friction
            last_v_n_i = (vm1m0+vm1p1)*p5
            v_n = (vm0m0+vm0p1)*p5  # stradles n[i,j]
            # this calculation is different that the one used before because it
            # is u-centered directly rather then averaging it.
            # malfuncitons if i=0 or i = n
            last_depth_x = nm1m0+hm1m0  
            depth = nm0m0+hm0m0
            depth_u = (depth+last_depth_x)*p5  # straddles u[i]
            v_u =   (last_v_n_i +v_n)*p5  # straddles u[i]
            u1 = u[i,j]
            ro = np.float32(1)  # remove
            drag = -mu*u1*math.sqrt(u1*u1+ v_u*v_u)/(ro*depth_u)

            k = drag +advection+grav+coriolis_u #grav+advection+coriolis_u+drag
            print("cu_shared",jj,ii)
            out_u[ii,jj]= ii+jj/100.0 #k    
# cuda.syncthreads()  # maybe not needed
cu_u_driver_shared = compileit(cu_u_driver_shared_py)
#cu_u_driver_shared.__name__="cu_u_driver_shared"
#cu_u_driver_shared.BLOCKDIM  = (threadsperblock[0],threadsperblock[1])
#@cuda.jit('void(int32,int32,int32,int32,float32[:,:], float32[:,:], float32[:,:],float32[:,:],float32[:,:],float32,float32,float32[:,:])',device=True)

sqrt = math.sqrt

def dudt2_drag_ret_py(i,j,h, n, f, u, v, dx, dy) : 
    p5 = np.float32(0.5)
# physics constants
#attenuation by friction
    last_v_n_i = (v[i-1,j]+v[i-1,j+1])*p5
    v_n = (v[i,j]+v[i,j+1])*p5  # stradles n[i,j]
    # this calculation is different that the one used before because it
    # is u-centered directly rather then averaging it.
    # malfuncitons if i=0 or i = n
    #last_depth_x = n[i-1,j]+h[i-1,j]  

     
    depth =  n[i,j] +h[i,j]
    last_depth_x = h[i-1,j] +n[i-1,j] #
    
    depth_u =  np.float32(1E-9)+abs(depth+last_depth_x)*p5  # straddles u[i]
    

    v_u =   (last_v_n_i +v_n)*p5  # straddles u[i]
    uu = u[i,j]
    ro = np.float32(1)  # remove
    drag = -mu* uu*sqrt(uu*uu+ v_u*v_u)/(ro*depth_u)
    return drag
    
def dudt2_coriolis_ret_py(i,j,h, n, f, u, v, dx, dy) :
    p5 = np.float32(0.5)
    # first average v along j axis to make it n-centered
    #v_n[i,j] = (v[i,j]+v[i,j+1])*p5  # average neighbors. #### v_n is n-centered between v'000000s
    last_v_n_i = (v[i-1,j]+v[i-1,j+1])*p5
    v_n = (v[i,j]+v[i,j+1])*p5  # stradles n[i,j]

    coriolis_u = (f[i-1,j]*last_v_n_i+f[i,j]*v_n)*p5 # coriolis force F is n-centered
    return coriolis_u

def dudt2_dummy_ret_py(i,j,h, n, f, u, v, dx, dy) :
    p5 = np.float32(0.5)
    # first average v along j axis to make it n-centered
    #v_n[i,j] = (v[i,j]+v[i,j+1])*p5  # average neighbors. #### v_n is n-centered between v'000000s
  #  last_v_n_i = (v[i-1,j]+v[i-1,j+1])*p5
    v_n = (v[i,j]+v[i,j+1])*p5  # stradles n[i,j]

    coriolis_u = (f[i,j]*v_n)*p5 # coriolis force F is n-centered
    return coriolis_u


def dudt2_grav_ret_py(i,j,h, n, f, u, v, dx, dy):
# physics constants
    g = np.float32(9.81) # gravity
    # gravity  (is this wrong in main routine== check indexing)
    # malfunction if i=0 or i=n 
    grav = (n[i-1,j]-n[i,j])*(g)/dx  #n[i-1] and n[i] straddle u[i]
    return grav

def dudt2_advection_ret_py(i,j,h, n, f, u, v, dx, dy) : 
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
        


device_ret_compiler = cuda.jit(  'void(int32,int32,int32,int32,float32[:,:], float32[:,:], float32[:,:],float32[:,:],float32[:,:],float32,float32,float32[:,:])',device=True,fastmath=True) 
device_ii_compiler = cuda.jit('float32(int32,int32,float32[:,:], float32[:,:], float32[:,:],float32[:,:],float32[:,:],float32,float32)',device=True,inline=False,fastmath=True) 
cuda_dudt2_grav_ret = device_ii_compiler(dudt2_grav_py)
cuda_dudt2_advection_ret = device_ii_compiler(dudt2_advection_py)
cuda_dudt2_drag_ret = device_ii_compiler(dudt2_drag_py)
cuda_dudt2_coriolis_ret = device_ii_compiler(dudt2_coriolis_py)
cuda_dudt2_dummy_ret = device_ii_compiler(dudt2_dummy_py)  ####################
cuda_dvdt2_grav_ret = device_ii_compiler(dvdt2_grav_py)
cuda_dvdt2_advection_ret = device_ii_compiler(dvdt2_advection_py)
cuda_dvdt2_drag_ret = device_ii_compiler(dvdt2_drag_py)
cuda_dvdt2_coriolis_ret = device_ii_compiler(dvdt2_coriolis_py)
device_ii_ret_compiler = cuda.jit('float32(int32,int32,float32[:,:], float32[:,:], float32[:,:],float32[:,:],float32[:,:],float32,float32)',device=True) 

funcs = device_ii_ret_compiler(funcs)
inner = device_ret_compiler(inner)
#device_compiler = cuda.jit('float32(int32,int32,float32[:,:], float32[:,:], float32[:,:],float32[:,:],float32[:,:],float32,float32)',device=True) 
#cuda_dudt2_grav=device_compiler(dudt2_grav_py)
#cuda_dudt2_advection=device_compiler(dudt2_advection_py)

#cuda_dudt2_coriolis=device_compiler(dudt2_coriolis_py)
#cuda_dudt2_drag=device_compiler(dudt2_drag_py)

#@cuda.jit('void(float32[:,:], float32[:,:], float32[:,:],float32[:,:],float32[:,:],float32,float32,float32[:,:])', inline=True)
def cu_u_driver_shared_inner_py(uh, un, uu, uv, uf, dx, dy, out_u):
    """This kernel function will be executed by a thread."""
    jj,ii  = cuda.grid(2)  # slow,fast      ii,jj  = cuda.grid(2)   #slow,fast    # in our system i is slow/far and j is fast/near,  usually defined differently
                                          # cuda  y is slow and cuda  x is fast  (this is also flipped from our system)
                                          #  in our system    i(x) --> cuda.y  and j(y) --> cuda.x
                                          #  i is axis[0] and j is axis[1] in our system.  flipped from normal
#THIS NEED FIXING
    # Allocate static shared memory of 512 (max number of threads per block for CC < 3.0)
    # This limits the maximum block size to 512.
#    sh = cuda.shared.array(shape=(TPB_slow,TPB_fast), dtype=nb.float32)
#    sn = cuda.shared.array(shape=(TPB_slow,TPB_fast), dtype=nb.float32)
    su = cuda.shared.array(shape=(TPB_slow,TPB_fast), dtype=nb.float32) #33
    sv = cuda.shared.array(shape=(TPB_slow,TPB_fast), dtype=nb.float32) # 33
#    sf = cuda.shared.array(shape=(TPB_slow,TPB_fast), dtype=nb.float32)

    j = cuda.threadIdx.x
    i = cuda.threadIdx.y
    by = cuda.blockDim.y
    bx = cuda.blockDim.x
#    i = tx + bx * bw
    if ii < uu.shape[0] and jj<uu.shape[1]:
        su[i,j] = uu[ii,jj]                     # I think this is strding wrong
#     if ii < un.shape[0]-1 and jj<un.shape[1]:
#          sh[i,j] = uh[ii,jj]
#          sn[i,j] = un[ii,jj]    
#          sf[i,j] = uf[ii,jj]
    if ii < uv.shape[1] and jj<uv.shape[0]:
        sv[i,j] = uv[ii,jj]
    
    n =un #sn
    h =uh #sh
    u = su
    v = sv
    f = uf #sf
    
#     h = uh
#     n = un
#     u = uu
#     v = uv
#     f = uf
#      i,ii = ii,i
#      j,jj = jj,j
#    i = ii
#    j = jj
    cuda.syncthreads() # wait for shared memory fill to complete

    if (ii < out_u.shape[1]-1) and (jj < out_u.shape[0]):
        k = np.float32(0.0)
        if  i>0  and j>0  and i< by-2 and j<bx-2 : #####################################################
            #out_u[ii,jj] =  #cuda_dudt2_grav_ret(ii,jj,i,j,h, n, f, u, v, dx, dy) 
           inner(ii,jj,i,j,h, n, u, v, f, dx, dy, out_u)
   # cuda.syncthreads()  # maybe not needed
cu_u_driver_shared_inner=compileit(cu_u_driver_shared_inner_py)
cu_u_driver_shared_inner.__name__="cu_u_driver_shared_inner"
cu_u_driver_shared_inner.BLOCKDIM  = (threadsperblock[0],threadsperblock[1])


def make_cuda_u():
    fns = []
    s = 'float32,float32,float32[:,:], float32[:,:], float32[:,:],float32[:,:],float32[:,:],float32,float32'
    for fn in (dudt2_drag_py,dudt2_advection_py,dudt2_grav_py,dudt2_coriolis_py):

        name = fn.__name__
        print ("name", name)
        x = nb.cuda.jit(fn,'float32('+s+')',device=True)
        fns.append(x)
    def dudt2_all_py(i,j,h, n, f, u, v, dx, dy) : 
        dudt=np.float32(0.0)
        for x in fns:
            dudt += x(i,j,h, n, f, u, v, dx, dy)  
        return dudt
    dudt2_all = nb.cuda.jit(dudt2_all_py,'float32('+s+')',device=True)




@cuda.jit('void(int32,int32,int32,int32,float32[:,:],float32[:,:])',device=True)  
def cuda_load_shared(gi,gj,si,sj,su,uu):
    # this loads everything but the edges along the limit cases 
    # the reason we don't load these it it requires a threadsync and 
    # we want to minimize that.
    #    sj = cuda.threadIdx.x;     
  #  si = cuda.threadIdx.y;
#     if si > threadsperblock[1]-1 or sj > threadsperblock[0]-1:   
#         print ("s index violation", si,sj, shape=(TPB_slow,TPB_fast), "idx",cuda.threadIdx.x,"idy",cuda.threadIdx.y)
#     if ( gj>uu.shape[1]-1 ) : 
#         print ("gj index viloation",gi,gj,cuda.blockIdx.x,cuda.blockDim.x,cuda.threadIdx.x,cuda.blockIdx.x*(cuda.blockDim.x-2))
#     if ( gi>uu.shape[0]-1 ) : 
#         print ("gi index viloation",uu.shape[0],si,sj,"gij",gi,gj,"indx",cuda.blockIdx.y,cuda.blockDim.y,cuda.threadIdx.y,cuda.blockIdx.y*(cuda.blockDim.y-2))
#     
    if gi < uu.shape[0] and gj<uu.shape[1] and gi>-1 and gj>-1:
        su[si,sj] = uu[gi,gj]

@cuda.jit('void(int32,int32,int32,int32,float32[:,:],float32[:,:])',device=True)  
def cuda_load_shared_corners(gi,gj,si,sj,su,uu):
    # don't even need to enter this if in the interior.
    im,jm = uu.shape    
    # do the edges
    if gi == -1 and -1<gj<jm:
        su[0,sj] = su[1,sj]
    elif gj == -1 and -1<gi<im:    # elif collision at corer case 
        su[si,0] = su[si,1]
        
    if gi == im and -1<gj<jm:
        su[si,sj] = su[si-1,sj] 
    elif gj == jm and -1<gi<im: 
        su[si,sj] = su[si,sj-1]
        
    # corners at limits    
    if gi==-1 and gj==-1:
        su[0,0] = su[1,1]    
    elif gi==im and gj==jm:
        su[si,sj]=su[si-1,sj-1]
    elif gi==im and gj==-1:
        su[si,0]= su[si-1,0]
    elif gi == -1 and gj==jm:
        su[0,sj]= su[0,sj-1]

  
def cuda_u_driver_shared_py(uh, un, uu, uv, uf, dx, dy, out_u):
    """This kernel function will be executed by a thread."""

   # THE LOGIC OF i and j IS REVERSED IN THIS VERSION.
  #ii,jj  = cuda.grid(2)   #slow,fast    # in our system i is slow/far and j is fast/near,  usually defined differently
                                          # cuda  y is slow and cuda  x is fast  (this is also flipped from our system)
                                          #  in our system    i(x) --> cuda.y  and j(y) --> cuda.x
                                          #  i is axis[0] and j is axis[1] in our system.  flipped from normal

    gj  = cuda.blockIdx.x*(cuda.blockDim.x-2)+cuda.threadIdx.x-1;  # could leave off the -1 if need be.
    gi  = cuda.blockIdx.y*(cuda.blockDim.y-2) + cuda.threadIdx.y-1;
    j  = cuda.threadIdx.x;
    i = cuda.threadIdx.y;
    sj = cuda.threadIdx.x;     
    si = cuda.threadIdx.y; 


  #  Allocate static shared memory of 512 (max number of threads per block for CC < 3.0)
  #  This limits the maximum block size to 512.
    sh = cuda.shared.array(shape=(TPB_slow,TPB_fast), dtype=nb.float32)
    su = cuda.shared.array(shape=(TPB_slow,TPB_fast), dtype=nb.float32) #33
    sv = cuda.shared.array(shape=(TPB_slow,TPB_fast), dtype=nb.float32) # 33
    sf = cuda.shared.array(shape=(TPB_slow,TPB_fast), dtype=nb.float32)
    sn = cuda.shared.array(shape=(TPB_slow,TPB_fast), dtype=nb.float32)
    

    cuda_load_shared(gi,gj,si,sj,su,uu)
    cuda_load_shared(gi,gj,si,sj,sv,uv)
    cuda_load_shared(gi,gj,si,sj,sn,un)
    cuda_load_shared(gi,gj,si,sj,sh,uh)
    cuda_load_shared(gi,gj,si,sj,sf,uf)
    
    cuda.syncthreads()  # require threadsync before filling in corners.
    
    if (cuda.blockIdx.x ==0 or cuda.blockIdx.y==0 or cuda.blockIdx.y==cuda.blockDim.y-1 or cuda.blockIdx.x==cuda.blockDim.x-1 ) :
    
        cuda_load_shared_corners(gi,gj,si,sj,sf,uf)
        cuda_load_shared_corners(gi,gj,si,sj,sh,uh)
        cuda_load_shared_corners(gi,gj,si,sj,sn,un)
        cuda_load_shared_corners(gi,gj,si,sj,sv,uv)
        cuda_load_shared_corners(gi,gj,si,sj,su,uu)
 
        cuda.syncthreads() # finalize the shared memory before going on.
            
    n = sn #un #sn
    h = sh # uh #sh
    u = su
    v = sv
    f = sf #uf #sf
    # discard edge threads and truncate at boundaries

    if cuda.blockDim.y-1>si>0 and cuda.blockDim.x-1>sj>0 and gi<out_u.shape[0] and gj<out_u.shape[1] :
        tmp=np.float32(0.0)
        tmp +=  cuda_dudt2_drag_ret(si,sj,h, n, f, u, v, dx, dy)  
        tmp += cuda_dudt2_advection_ret(si,sj,h, n, f, u, v, dx, dy)  
        tmp += cuda_dudt2_grav_ret(si,sj,h, n, f, u, v, dx, dy) 
        tmp += cuda_dudt2_coriolis_ret(si,sj,h, n, f, u, v, dx, dy)
          
    #    tmp += cuda_dvdt2_drag_ret(sj,si,h, n, f, u, v, dx, dy)  
    #    tmp += cuda_dvdt2_advection_ret(sj,si,h, n, f, u, v, dx, dy)  
    #    tmp += cuda_dvdt2_grav_ret(sj,si,h, n, f, u, v, dx, dy) 
    #    tmp += cuda_dvdt2_coriolis_ret(sj,si,h, n, f, u, v, dx, dy)  
        out_u[gi,gj] = tmp # gi+gj/100.0 #tmp  ##  IS THIS REGISTERED RIGHT??  SHOULD IT be gj+1, gi+1:  I think so because si and sj are restricted.

    
def cuda_u_driver_shared_inner_py(uh, un, uu, uv, uf, dx, dy, out_u):
    """This kernel function will be executed by a thread."""

   # THE LOGIC OF i and j IS REVERSED IN THIS VERSION.
      # in our system i is slow/far and j is fast/near,  usually defined differently
                                          # cuda  y is slow and cuda  x is fast  (this is also flipped from our system)
                                          #  in our system    i(x) --> cuda.y  and j(y) --> cuda.x
                                          #  i is axis[0] and j is axis[1] in our system.  flipped from normal
   # ii,jj  = cuda.grid(2)   #cuda.x (fast), cuda.y (slow) 
    
    gj  = cuda.blockIdx.x*(cuda.blockDim.x-2)+cuda.threadIdx.x-1;  # could leave off the -1 if need be.
    gi  = cuda.blockIdx.y*(cuda.blockDim.y-2) + cuda.threadIdx.y-1;
    j  = cuda.threadIdx.x;
    i = cuda.threadIdx.y;
    sj = cuda.threadIdx.x;     
    si = cuda.threadIdx.y; 

  #  Allocate static shared memory of 512 (max number of threads per block for CC < 3.0)
  #  This limits the maximum block size to 512.
#    sh = cuda.shared.array(shape=(TPB_slow,TPB_fast), dtype=nb.float32)
    su = cuda.shared.array(shape=(TPB_slow,TPB_fast), dtype=nb.float32) #33
    sv = cuda.shared.array(shape=(TPB_slow,TPB_fast), dtype=nb.float32) # 33
#    sf = cuda.shared.array(shape=(TPB_slow,TPB_fast), dtype=nb.float32)
#    sn = cuda.shared.array(shape=(TPB_slow,TPB_fast), dtype=nb.float32)

    cuda_load_shared(gi,gj,si,sj,su,uu)
    cuda_load_shared(gi,gj,si,sj,sv,uv)
#     cuda_load_shared(gi,gj,si,sj,sn,un)
#     cuda_load_shared(gi,gj,si,sj,sh,uh)
#     cuda_load_shared(gi,gj,si,sj,sf,uf)
    
    cuda.syncthreads()  # require threadsync before filling in corners.
    
    if (cuda.blockIdx.x ==0 or cuda.blockIdx.y==0 or cuda.blockIdx.y==cuda.blockDim.y-1 or cuda.blockIdx.x==cuda.blockDim.x-1 ) :
#         cuda_load_shared_corners(gi,gj,si,sj,sf,uf)
#         cuda_load_shared_corners(gi,gj,si,sj,sh,uh)
#         cuda_load_shared_corners(gi,gj,si,sj,sn,un)
        cuda_load_shared_corners(gi,gj,si,sj,sv,uv)
        cuda_load_shared_corners(gi,gj,si,sj,su,uu)
 
        cuda.syncthreads() # finalize the shared memory before going on.
            
    n = un #sn
    h = uh #sh
    u = su
    v = sv
    f = uf #sf
    # discard edge threads and truncate at boundaries
    
    # note that the -1 on the ji and gj checks is overkill.  we currently don't fix the plus1 on N in the last row.
    #  if we checked for that we could extend this fill.
 
    if cuda.blockDim.y-1>si>0 and cuda.blockDim.x-1>sj>0 and gi<out_u.shape[0]-1 and gj<out_u.shape[1]-1 :
        inner(gi,gj,si,sj,h, n, u, v, f, dx, dy, out_u)


    
cuda_u_driver_shared= compileit(cuda_u_driver_shared_py,blockdim=threadsperblock,border=(2,2)) 
cuda_u_driver_shared_inner  = compileit(cuda_u_driver_shared_inner_py,blockdim=threadsperblock,border=(2,2))
cu_u_driver_global_inner=compileit(cu_u_driver_global_inner_py)
cu_u_driver_global_funcs=compileit(cu_u_driver_global_funcs_py)
cu_u_driver_global_inner_col = compileit(cu_u_driver_global_inner_col_py,innerloop=True)
cu_u_driver_global_inner_col.BLOCKDIM = (cu_u_driver_global_inner_col._func.get().attrs.maxthreads,1)
cu_u_driver_global_funcs_col = compileit(cu_u_driver_global_funcs_col_py,innerloop=True)
cu_u_driver_global_funcs_col.BLOCKDIM = (cu_u_driver_global_funcs_col._func.get().attrs.maxthreads,1)
#cuda_u_driver_shared= compiler(cuda_u_driver_shared_py)  # this is stupidly slow!  why?
#cuda_u_driver_shared.__name__='cuda_u_driver_shared'
# cuda_u_driver_shared.BLOCKDIM  = (threadsperblock[0],threadsperblock[1])
# cuda_u_driver_shared.BORDER = (2,2)
# 
# cuda_u_driver_shared_inner  = compiler(cuda_u_driver_shared_inner_py)
# cuda_u_driver_shared_inner.__name__='cuda_u_driver_shared_inner'
# cuda_u_driver_shared_inner.BLOCKDIM  = (threadsperblock[0],threadsperblock[1])
# cuda_u_driver_shared_inner.BORDER = (2,2)
# 
# 
# cu_u_driver_global_inner=compiler(cu_u_driver_global_inner_py)
# cu_u_driver_global_inner.__name__="cu_u_driver_global_inner"
# cu_u_driver_global_inner.BLOCKDIM  = threadsperblock  # slow fast
# 
# cu_u_driver_global_funcs=compiler(cu_u_driver_global_funcs_py)
# cu_u_driver_global_funcs.__name__="cu_u_driver_global_funcs"
# cu_u_driver_global_funcs.BLOCKDIM = threadsperblock


    
    

mytime = time.perf_counter
#mytime = time.time
REPS = 3
ITRS = 2            
def testin():
    N=10000
    M=1000
    h = np.asarray( np.float32(2)+np.random.random((N,M)) ,dtype=np.float32)
    n = np.asarray(np.float32(2)+np.random.random((N,M)) ,dtype=np.float32)
    u = np.asarray(np.random.random((N+1,M)) ,dtype=np.float32)
    v = np.asarray(np.random.random((N,M+1)) ,dtype=np.float32)
    f = np.asarray(np.random.random((N,M)) ,dtype=np.float32)
    dx = np.float32(0.1)
    dy = np.float32(0.2)
    #p.g = np.float32(1.0)
    nu= np.float32(1.0)
    



    out_u = np.asarray(np.random.random(u.shape) ,dtype=np.float32)
    out_v = np.asarray(np.random.random(v.shape) ,dtype=np.float32)

    old_u = np.asarray(np.zeros(u.shape) ,dtype=np.float32)
    old_v = np.asarray(np.zeros(v.shape) ,dtype=np.float32)
    
    timings = {}
    

    
    def excersize(cu_u_driver,res1=None ):
                # since blocks can be overlapped with walk the number of
                # blocks we need has to take the overlap borders into account.
                etpb_fast,etpb_slow = (cu_u_driver.BLOCKDIM[0]-cu_u_driver.BORDER[0],(cu_u_driver.BLOCKDIM[1]-cu_u_driver.BORDER[1])) # (fast,slow)
                
                    
                blockspergrid_fast = (u.shape[1] + etpb_fast) // etpb_fast  # fast  
                blockspergrid_slow = (u.shape[0] + etpb_slow) // etpb_slow # slow
                if cu_u_driver.INNERLOOP: blockspergrid_slow=1
                   
               # blockspergrid = (blockspergrid_slow, blockspergrid_fast) # blocks along slow, blocks along fast
                blockspergrid = (blockspergrid_fast, blockspergrid_slow) # blocks along slow, blocks along fast
                threadsperblock = (cu_u_driver.BLOCKDIM[0],cu_u_driver.BLOCKDIM[1])
                print ("threadsperblock[0],threadsperblock[1],etpb",threadsperblock[0],threadsperblock[1],etpb_fast,etpb_slow)

                      # notice this is different than the threadblock order.
                      #  tthreadsperblock (Blockdim) ( block_length_in_fast_direction , block_length_in_slow_direction) 
                      #  blockspergrid (griddim)( matrix_length_in_slow_direction   /Block_length_in_slow_dir, Matrix_length_in_fast_direction/Block_lenghth_inFast_dir )
                return testit(blockspergrid,threadsperblock,cu_u_driver,res1)
                 
    def testit(blockspergrid,threadsperblock,cu_u_driver,res1=None ):
                print (cu_u_driver.__name__)
                h1 = cuda.to_device(h)
                n1 = cuda.to_device(n)
                u1 = cuda.to_device(u)
                v1 = cuda.to_device(v)
                f1 = cuda.to_device(f)
                out_u[:,:]= 0.0
                out_u1 =cuda.to_device(out_u)
                
    
            
                print( "blocks per grid", blockspergrid)
                print("threads per block",threadsperblock)
                print(cu_u_driver._func.get().attrs)
            
                ts = []
                for i in range(REPS):
      
                    t = mytime()  # time.process_time()
                    for j in range(ITRS):   
                        cu_u_driver[blockspergrid,threadsperblock](h1, n1, u1, v1, f1, dx, dy, out_u1)
                        cu_u_driver[blockspergrid,threadsperblock](h1, n1, u1, v1, f1, dx, dy, out_u1)
                        cu_u_driver[blockspergrid,threadsperblock](h1, n1, u1, v1, f1, dx, dy, out_u1)
           
                        cuda.synchronize()
                    t2 =  mytime()  # time.process_time()
                    ts.append(t-t2)
                print("cuda timing") 
                print (np.median(ts),np.min(ts),np.max(ts),np.std(ts))
                timings[cu_u_driver.__name__+str(cu_u_driver.BLOCKDIM)]=np.median(ts)/ITRS/3
                print (ts)
                local_out = out_u1.copy_to_host()
                #print (local_out)
                if local_out.shape[0] < 20:
                    print ("full matrix")
                    for m1 in range(local_out.shape[0]):
                        for m2 in range(local_out.shape[1]):  print ("%7.2f "%(local_out[m1,m2],),end="")
                        print ()
                    print()    
                if res1 is None :  
                    res1 = local_out
                result =local_out-res1
                result = result[2:-2,2:-2]
                print ("segment")
                print(result [segment[0]:segment[1],segment[0]:segment[1]])
                print ("max %f min %f std %f"%(np.max(result),np.min(result),np.std(result)))
                jj = np.unravel_index(np.argmax(result ,axis=None), result.shape)
                print ("argmax",jj)
                print (result[jj[0]-4:jj[0]+4,jj[1]-4:jj[1]+4])
                return res1
                


 #   threadsperblock = (TPBX,TPBY) # (16,16)  # backwards to Cmatrix order!  needs to be fast.slow or x,y
##  dim3 dimGrid(nx/TILE_DIM, ny/TILE_DIM, 1);   Matrix_length_in_fast_direction/Block_lenghth_inFast_dir , matrix_length_in_slow_direction   /Block_length_in_slow_dir
#  dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);     ( block_length_in_fast_direction , block_length_in_slow_direction) 
#     blockspergrid_y = (u.shape[0] + threadsperblock[0]) // threadsperblock[0]
#     blockspergrid_x = (u.shape[1] + threadsperblock[1]) // threadsperblock[1]
#     blockspergrid = (blockspergrid_y, blockspergrid_x)
# 
#     print ("here we go",u.shape)
#     print( "blocks per grid", blockspergrid)
#     print("threads per block",threadsperblock)
    print ("here we go",u.shape)    
    segment =(u.shape[0]//10,u.shape[0]//10+5)
    res1=None
    try:
       # testit((1,1),(1,1001),cu_u_driver_global_funcs_col,res1)
        for kk,cu_u_driver in enumerate((cu_u_driver_global_inner,cu_u_driver_global_funcs,cu_u_driver_global_inner_col,cu_u_driver_global_funcs_col,cuda_u_driver_shared_inner,cu_u_driver_global,cu_u_driver_global_inner,cuda_u_driver_shared,)): #cu_u_driver_shared,cu_u_driver_shared_inner)):
            res1 = excersize(cu_u_driver,res1=res1 )
        #cu_u_driver_global.BLOCKDIM = (16,32)
       # res1 = excersize(cu_u_driver_global,res1=res1 )
       # cu_u_driver_global.BLOCKDIM = (33,15)
       # res1 = excersize(cu_u_driver_global,res1=res1 ) 
       # cu_u_driver_global_inner.BLOCKDIM = (1024,1)
       # cu_u_driver_global_funcs.BLOCKDIM = (1,1024)
        
       # testit((1,1),(1,1001),cu_u_driver_global_funcs_col,res1)
        
      #  for kk,cu_u_driver in enumerate((cu_u_driver_global_inner,cu_u_driver_global_funcs,cu_u_driver_global)):
      #           res1 = excersize(cu_u_driver,res1=res1 )  
#             
            
    finally:
        print ("cuda closer")
        cuda.close()  

  
    print ("all done")
   # %timeit forward2_u_driver (h, n, u, v,  dx, dy, out_u)
   # %timeit   orig_v = dudt(h, n, f, u, v, dx, dy, nu )
    ts = []
    out_u = np.asarray(np.zeros((N+1,M)) ,dtype=np.float32)
    for i in range(REPS):
        t = mytime()  # time.process_time()
        for j in range(ITRS):
            forward2_u_driver(h, n, u, v, f, dx, dy, out_u)
            forward2_u_driver(h, n, u, v, f, dx, dy, out_u)
            forward2_u_driver(h, n, u, v, f, dx, dy, out_u)
        t2 =  mytime()  # time.process_time()
        ts.append(t-t2)
    print("numba timing")
    print (np.median(ts),np.min(ts),np.max(ts),np.std(ts))
    timings['numba_u']=np.median(ts)/ITRS/3
    result =out_u-res1
    result = result[2:-2,2:-2]
    print ("segment")
    print(result [segment[0]:segment[1],segment[0]:segment[1]])
    print ("max %f min %f std %f"%(np.max(result),np.min(result),np.std(result)))
    jj = np.unravel_index(np.argmax(result ,axis=None), result.shape)
    print("argmax",jj)
    print (result[jj[0]-4:jj[0]+4,jj[1]-4:jj[1]+4])
   # print (res1[2:-1,2:-2][jj[0]-4:jj[0]+4,jj[1]-4:jj[1]+4])
   # print (out_u[2:-2,2:-2]  [jj[0]-4:jj[0]+4,jj[1]-4:jj[1]+4])
    #print(out_u[segment[0]:segment[1],segment[0]:segment[1]])

    ts = []
    old_u = np.asarray(np.zeros((N+1,M)) ,dtype=np.float32)
    print(dx,dy)
    for i in range(REPS):
        
        t = mytime()  # time.process_time()
        for j in range(ITRS):
            dudt_all(h=h, n=n, f=f, u=u, v=v, dx=dx, dy=dy,out=old_u)
            dudt_all(h=h, n=n, f=f, u=u, v=v, dx=dx, dy=dy,out=old_u)
            dudt_all(h=h, n=n, f=f, u=u, v=v, dx=dx, dy=dy,out=old_u)
        t2 =  mytime()  # time.process_time()
        ts.append(t-t2)

    print("dudt_all")         
    print (np.median(ts),np.min(ts),np.max(ts),np.std(ts))    
    old_u = np.asarray(np.zeros((N+1,M)) ,dtype=np.float32) 
    dudt_all(h=h, n=n, f=f, u=u, v=v, dx=dx, dy=dy,out=old_u)
    result =old_u-res1
    result = result[2:-2,2:-2]
    print ("segment")
    print(result[segment[0]:segment[1],segment[0]:segment[1]])   
    print ("max %f min %f std %f"%(np.max(result),np.min(result),np.std(result)))
    jj = np.unravel_index(np.argmax(result ,axis=None), result.shape)
    print("argmax",jj)
    print (result[jj[0]-4:jj[0]+4,jj[1]-4:jj[1]+4])
    ts = []
    for i in range(REPS):
        t = mytime()  # time.process_time()
        for j in range(ITRS):    
            orig_v = dudt(h, n, f, u, v, dx, dy)
            orig_v = dudt(h, n, f, u, v, dx, dy)
            orig_v = dudt(h, n, f, u, v, dx, dy)
        t2 =  mytime()  # time.process_time()
        ts.append(t-t2)
    print("orig") 
    print (np.median(ts),np.min(ts),np.max(ts),np.std(ts))
    timings['dudt_all']=np.median(ts)/ITRS/3
    print(ts)
    result = orig_v-res1
    result = result[2:-2,2:-2]
    print ("segment")
    print(result[segment[0]:segment[1],segment[0]:segment[1]])   
    print ("max %f min %f std %f"%(np.max(result),np.min(result),np.std(result)))
    jj = np.unravel_index(np.argmax(result ,axis=None), result.shape)
    print("argmax",jj)
    print (result[jj[0]-4:jj[0]+4,jj[1]-4:jj[1]+4])
    
    
    out_v = np.asarray(np.zeros((N,M+1)) ,dtype=np.float32)
    ts=[]
    for i in range(REPS):
        t = mytime()  # time.process_time()
        for j in range(ITRS):
            grid_driver_v(h, n, u, v, f, dx, dy, out_v)
            grid_driver_v(h, n, u, v, f, dx, dy, out_v)
            grid_driver_v(h, n, u, v, f, dx, dy, out_v)
        t2 =  mytime()  # time.process_time()
        ts.append(t-t2)

    print("numba v grid")
    print (np.median(ts),np.min(ts),np.max(ts),np.std(ts))
    timings['numba_v_grid']=np.median(ts)/ITRS/3
    ts=[]
    for i in range(REPS):
        t = mytime()  # time.process_time()
        for j in range(ITRS):
            forward2_v_driver (h, n, u, v, f,dx, dy, out_v)
            forward2_v_driver (h, n, u, v, f, dx, dy, out_v)
            forward2_v_driver (h, n, u, v, f, dx, dy, out_v)
        t2 =  mytime()  # time.process_time()
        ts.append(t-t2)
    print("numba v")
    print (np.median(ts),np.min(ts),np.max(ts),np.std(ts))
    timings['numba_v']=np.median(ts)/ITRS/3
    ts = []
    for i in range(REPS):
        t = mytime()  # time.process_time()
        for j in range(ITRS):
            orig_v = dvdt(h, n, f, u, v, dx, dy, nu, mu=np.float32(0.3) )
            orig_v = dvdt(h, n, f, u, v, dx, dy, nu, mu=np.float32(0.3) )
            orig_v = dvdt(h, n, f, u, v, dx, dy, nu, mu=np.float32(0.3) )
        t2 =  mytime()  # time.process_time()
        ts.append(t-t2)
    print("orig") 
    print (np.median(ts),np.min(ts),np.max(ts),np.std(ts))

#     ts = []
#     old_v = np.asarray(np.zeros((N,M+1)) ,dtype=np.float32)
#     for i in range(10):
#         t = mytime()  # mytime()  # time.process_time()
#         for j in range(4):
#             dvdt_orig_mimic (h=h, n=n, f=f, u=u, v=v, dx=dx, dy=dy,out=old_v)
#             dvdt_orig_mimic (h=h, n=n, f=f, u=u, v=v, dx=dx, dy=dy,out=old_v)
#             dvdt_orig_mimic (h=h, n=n, f=f, u=u, v=v, dx=dx, dy=dy,out=old_v)
#         t2 =  mytime()  # mytime()  # time.process_time()
#         ts.append(t-t2)
#     print("dvdt_mimic") 
#     print (np.median(ts),np.min(ts),np.max(ts),np.std(ts))
#     old_v = np.asarray(np.zeros((N,M+1)) ,dtype=np.float32)
    
    return timings,(N,M)

timing,(N,M) = testin()
print (timing)
tsort = np.sort([(timing[k],k) for k in timing])
print (tsort)
for k in timing:  
    print ("%9.7f\t%s"%(timing[k],k))
    
print (N,M)


# -0.0000844	cu_u_driver_global_inner_pycuda(32, 16)
# -0.0000965	cu_u_driver_global_funcs_pycuda(32, 16)
# -0.0000865	cuda_u_driver_shared_inner_pycuda(32, 16)
# -0.0000847	cu_u_driver_global_pycuda(32, 16)
# -0.0000922	cuda_u_driver_shared_pycuda(32, 16)
# -0.0000973	cu_u_driver_global_pycuda(16, 32)
# -0.0000852	cu_u_driver_global_pycuda(33, 15)
# -0.0001110	cu_u_driver_global_inner_pycuda(1024, 1)
# -0.0001104	cu_u_driver_global_funcs_pycuda(1, 1024)
# -0.0009721	numba_u
# -0.0020372	dudt_all
# -0.0013031	numba_v_grid
# -0.0012260	numba_v
# 1000 100

# -0.0001664	cu_u_driver_global_inner_pycuda(32, 16)
# -0.0001778	cu_u_driver_global_funcs_pycuda(32, 16)
# -0.0002158	cuda_u_driver_shared_inner_pycuda(32, 16)
# -0.0001764	cu_u_driver_global_pycuda(32, 16)
# -0.0003981	cuda_u_driver_shared_pycuda(32, 16)
# -0.0001900	cu_u_driver_global_pycuda(16, 32)
# -0.0002047	cu_u_driver_global_pycuda(33, 15)
# -0.0001868	cu_u_driver_global_inner_pycuda(1024, 1)
# -0.0007522	cu_u_driver_global_funcs_pycuda(1, 1024)
# -0.0084261	numba_u
# -0.0204474	dudt_all
# -0.0115580	numba_v_grid
# -0.0116596	numba_v
# 1000 1000

# -0.0013532	cu_u_driver_global_inner_pycuda(32, 16)
# -0.0014639	cu_u_driver_global_funcs_pycuda(32, 16)
# -0.0019563	cuda_u_driver_shared_inner_pycuda(32, 16)
# -0.0014868	cu_u_driver_global_pycuda(32, 16)
# -0.0036372	cuda_u_driver_shared_pycuda(32, 16)
# -0.0015526	cu_u_driver_global_pycuda(16, 32)
# -0.0015616	cu_u_driver_global_pycuda(33, 15)
# -0.0012898	cu_u_driver_global_inner_pycuda(1024, 1)
# -0.0056377	cu_u_driver_global_funcs_pycuda(1, 1024)
# -0.0684806	numba_u
# -0.3500182	dudt_all
# -0.0862844	numba_v_grid
# -0.0672100	numba_v
# 10000 1000

# -0.0115675	cu_u_driver_global_inner_pycuda(32, 16)
# -0.0122973	cu_u_driver_global_funcs_pycuda(32, 16)
# -0.0149882	cuda_u_driver_shared_inner_pycuda(32, 16)
# -0.0125394	cu_u_driver_global_pycuda(32, 16)
# -0.0264153	cuda_u_driver_shared_pycuda(32, 16)
# -0.0142666	cu_u_driver_global_pycuda(16, 32)
# -0.0125263	cu_u_driver_global_pycuda(33, 15)
# -0.0129114	cu_u_driver_global_inner_pycuda(1024, 1)
# -0.0580086	cu_u_driver_global_funcs_pycuda(1, 1024)
# -0.2315978	numba_u
# -3.1148582	dudt_all
# -0.3181006	numba_v_grid
# -0.2447898	numba_v
# 10000 10000
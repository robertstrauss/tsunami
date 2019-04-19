import argparse
import time
import pickle
from pathlib import Path

# wait until a runinfo.yml file exists
runinfo = Path("runinfo.dict")
while not runinfo.is_file():
    runinfo = Path("runinfo.dict")
    time.sleep(1)

# put status in output.html
s = '<!DOCTYPE-html>\n\n'

outfile = open('/home/rrs/shallowWater/www/html/output.html', 'w')# as outfile:
outfile.write(s + 'The simulation is running. This may take a few minutes. Output will be displayed here after a reload when finished.\n')


# read it and put values into local variables
with open("runinfo.dict", "rb") as pklfile:
    runinfodc = pickle.load(pklfile)
# latstart, latend, lonstart, lonend, tsulat, tsuamp, tsutype, tsuwx, tsuwy, theta = runinfodc['latstart'], runinfodc['latend'], runinfodc['lonstart'], runinfodc['lonend'], runinfodc['tsulat'], runinfodc['tsuamp'], runinfodc['tsutype'], runinfodc['tsuwx'], runinfodc['tsuwy'], runinfodc['theta']
try:
    latstart, latend, lonstart, lonend, tsulat, tsuamp, tsuwx, tsuwy, theta = \
    [float(runinfodc[k]) for k in 'latstart, latend, lonstart, lonend, tsulat, tsuamp, tsuwx, tsuwy, theta'.split(', ')]

except ValueError as e:
    print('error:')
    print(e)
    print('something was left blank, or an invalid value was submitted.')
    exit()
# vars = 'latstart, latend, lonstart, lonend, tsulat, tsuamp, tsutype, tsuwx, wsuwy, theta'.split(', ')
#
# parser = argparse.ArgumentParser(description='Run any part or all of my model.')
#
# for v in vars:
#     parser.add_argument(
#         '--'+v,
#         default=0,
#         type=np.float32,
#         help=v
#     )
# args = parser.parse_args()
# print(args)


# latstart, latend, lonstart, lonend, tsulat, tsuamp, tsutype, tsuwx, wsuwy, theta = args.latstart, args.latend, args.lonstart, args.lonend, args.tsulat, args.tsuamp, args.tsutype, args.tsuwx, args.wsuwy, args.theta




# load in the simulation framework
from shallowwater import *

outfile.write('loaded simulation framework.')

# set up initial condition
initcon = {}
latran = (latstart, latend) # latitude range map covers
lonran = (lonstart, lonend) # longitude range map covers

# calculate height of map  11100*lat degrees = meters
# calculate width of map  1 lon degree = cos(lat) lat degrees, *11100 = meters
# use lon degree size of average latitude
realsize = (111321*(latran[1]-latran[0]),\
            111321*(lonran[1]-lonran[0])*np.cos((latran[1]-latran[0])/2)*np.pi/180)# h, w of map in meters
aspect = realsize[0]/realsize[1]
size = (int(1000*aspect), int(1000/aspect))# grid size of the map lat, lon


initcon['dx'] = np.float32(realsize[1]/size[1])
initcon['dy'] = np.float32(realsize[0]/size[0])

# read in bathymetry data
bathdata = nc.Dataset('../data/bathymetry.nc','r')
bathlat = bathdata.variables['lat']
bathlon = bathdata.variables['lon']
#calculate indexes of bathymetry dataset we need
bathlatix = np.linspace(np.argmin(np.abs(bathlat[:]-latran[0])),\
                        np.argmin(np.abs(bathlat[:]-latran[1])),\
                        size[0], dtype=int)
bathlonix = np.linspace(np.argmin(np.abs(bathlon[:]-lonran[0])),\
                        np.argmin(np.abs(bathlon[:]-lonran[1])),\
                        size[1], dtype=int)
# print(bathlatix, bathlonix)
initcon['h'] = np.asarray(-bathdata.variables['elevation'][bathlatix, bathlonix], dtype=np.float32)
initcon['lat'] = np.asarray(bathlat[bathlatix])
initcon['lon'] = np.asarray(bathlon[bathlonix])


initcon['u'] = np.zeros((size[0]+1,size[1]), dtype=np.float32)
initcon['v'] = np.zeros((size[0],size[1]+1), dtype=np.float32)

initcon['dt'] = np.float32(0.2)*initcon['dx']/np.sqrt(np.max(initcon['h']*p.g))

# if tsutype == 'seismic':
#     initcon['n'] = seismic()
# else:
initcon['n'] = planegauss(size, tsuwx, tsuwy, int((tsulon-lonstart)*111321/initcon['dx']), \
                          int((tsulat-latstart)*111321*np.cos(latend*np.pi/180)/initcon['dy']), theta)

outfile.write('initial condition prepared. now simulating...\n')
ntt, maxn = simulate(initd, simtime, timestep=genfb, saveinterval=int(simtime/120), \
                     dudt_x = dudt_drive_numba, dvdt_x = dvdt_drive_numba, dndt_x = dndt_drive_numba, \
                     bounds=[0.97, 0.97, 0.97, 0.97])[:2]
outfile.write('finished simulating!\n')

mmax = np.max(maxn)/2

fig = plt.figure()

nttart = [(plt.imshow(nframe, cmap='seismic', vmin=-mmax, vmax=mmax),) for nframe in ntt]
anim = animation.ArtistAnimation(fig, nttart)

coast = plt.contour(initcon['h'], levels=1, colors='black')
xtixks = plt.xticks(np.linspace(0, initcon['h'].shape[1], 5),\
                    np.round(np.linspace(initcon['lon'][0], initcon['lon'][-1], 5), 3))
yticks = plt.yticks(np.linspace(0, initcon['h'].shape[0], 5),\
                    np.round(np.linspace(initcon['lat'][0], initcon['lat'][-1], 5), 3))
cb = plt.colorbar()
plt.savefig('anim')

outfile.write('created and saved animation. <a href="./anim.mp4">link to animation</a>')

fig = plt.figure()
plt.imshow(maxn, cmap='Reds')
coast = plt.contour(initcon['h'], levels=1, colors='black')
xtixks = plt.xticks(np.linspace(0, initcon['h'].shape[1], 5),\
                    np.round(np.linspace(initcon['lon'][0], initcon['lon'][-1], 5), 3))
yticks = plt.yticks(np.linspace(0, initcon['h'].shape[0], 5),\
                    np.round(np.linspace(initcon['lat'][0], initcon['lat'][-1], 5), 3))
cb = plt.colorbar()
plt.savefig('maxheights')

outfile.write('created and saved maximum water heights. <a href="./maxheights.png">link to maxheights</a>')

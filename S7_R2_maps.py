
"""
PURPOSE: PLOT R-SQUARED MAPS FOR BURNED AREA REGRESSIONS WITH SOIL MOISTURE AND PRECIPITATION AS THE PREDICTOR VARIABLE

Contact: Nathan Dadap, ndadap@stanford.edu
"""

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
import pandas as pd
from copy import copy
from matplotlib import gridspec
import seaborn as sns

#LOAD VARIABLE DATA
sm_rsquared_map = np.load('G:/My Drive/scripts/paper_plots/sm_ba/aic/sm_rsquared_map.npy')
precip_rsquared_map = np.load('G:/My Drive/scripts/paper_plots/sm_ba/aic/precip_rsquared_map.npy')
mean = np.load('G:/My Drive/scripts/paper_plots/sm_ba/aic/mean_ba.npy')

#PEATMASK
peat_area = np.load('G:/My Drive/data/peat_map/numpy_tables/peatarea_0.25deg.npy')
peat_area = peat_area[331:387,1097:1200]

########## PLOT

#SETUP PLOT LAYOUT
ftsz=24
fig=plt.figure(figsize=(9,10.3))
gs = gridspec.GridSpec(2, 2, height_ratios=[5,5], width_ratios=[100,1.5])
gs.update(wspace=.02,hspace=.1)

ax1=plt.subplot(gs[0,0])
ax2=plt.subplot(gs[1,0])
ax3=plt.subplot(gs[:,1])

#LOAD LAT AND LON VECTORS
import os
os.chdir('G:/My Drive/data/gfed_fire/gfed_numpy_tables/')
fire_lat=np.load('fire_lat.npy')
fire_lon=np.load('fire_lon.npy')
x,y=np.meshgrid(fire_lon,fire_lat)

#SETUP DATA FOR PLOTTING
plotter1 = np.zeros((720,1440))
plotter2 = np.zeros((720,1440))
pmsk = np.load('G:/My Drive/data/peat_map/numpy_tables/peatarea_0.25deg.npy')
plotter1[331:387,1097:1200] = pmsk[331:387,1097:1200]*sm_rsquared_map
plotter2[331:387,1097:1200] = pmsk[331:387,1097:1200]*precip_rsquared_map

# ADJUST CUSHION
m1 = Basemap(projection='cyl', llcrnrlat=-6, urcrnrlat=7,llcrnrlon=94.5, urcrnrlon=119, resolution='h', lon_0=0, ax=ax1)
m2 = Basemap(projection='cyl', llcrnrlat=-6, urcrnrlat=7,llcrnrlon=94.5, urcrnrlon=119, resolution='h', lon_0=0, ax=ax2)

# PLOT DATA ONTO MAP
cmap=copy(plt.cm.plasma)
cmap.set_bad('white', 1.)
cmap.set_over('yellow', 1.)

offset=.125
image1 = m1.pcolormesh(x-offset,y+offset,np.ma.masked_invalid(plotter1),shading='flat', cmap=cmap,vmin=0, vmax=1)
image2 = m2.pcolormesh(x-offset,y+offset,np.ma.masked_invalid(plotter2),shading='flat', cmap=cmap,vmin=0, vmax=1)


# DRAW COASTLINES AND MAP BOUNDARY
m1.drawcoastlines()
m1.drawmapboundary()
m2.drawcoastlines()
m2.drawmapboundary()

#COLORBAR
from matplotlib.colorbar import Colorbar
cb=Colorbar(ax=ax3, mappable = image1, orientation="vertical", ticks=[0,.25,.5,.75,1])             
cb.ax.tick_params(labelsize=ftsz-2)
cb.ax.set_ylabel('R$^2$ (BA$_{actual}$, BA$_{predicted}$)',fontsize=ftsz, labelpad=12)

#TEXT LABELS
ax1.text(.38,1.02,'Soil Moisture', fontsize=ftsz-2,transform=ax1.transAxes)
ax2.text(.38,1.02,'Precipitation', fontsize=ftsz-2, transform=ax2.transAxes)
#A and B label
ax1.text(-.05,.96,'a', weight='bold', fontsize=ftsz,transform=ax1.transAxes)
ax2.text(-.05,.96,'b', weight='bold',fontsize=ftsz, transform=ax2.transAxes)

plt.show()


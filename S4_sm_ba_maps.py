
"""
Purpose: Plot sample soil moisture and burned area map

Contact: Nathan Dadap, ndadap@stanford.edu
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.basemap import Basemap
from copy import copy

#variables: fire and sm (with lead time)
fire = np.load('G:/My Drive/data/gfed_fire/gfed_numpy_tables/fire_small_2015-2016_daily.npy')
firedates = pd.date_range('1/1/2015','12/31/2016', freq='D')
sm = np.load('G:/My Drive/data/smap_vod_sm/smap_numpy_tables/MTDCA_0.25deg_daily.npy')
smdates = pd.date_range('4/1/2015','3/31/2017', freq='D')

#for subsetting
latsize = 56
lonsize = 103

for lead in [10,20,30]:
    #get a copy of sm matrix, subset end of matrix and fill front of timeseries (based on lead time) with nans
    lag_var = np.concatenate((np.full((lead,latsize,lonsize),np.nan),sm[:(731-lead),:,:]),axis=0)
    lag_var[:lead] = np.nan
    if lead == 10:
        sm10 = lag_var
    if lead == 20:
        sm20 = lag_var
    if lead == 30:
        sm30 = lag_var

#initialize general start and end date time
start_date, end_date = pd.to_datetime('2014-01-01'), pd.to_datetime('2018-01-01')

#search through dates and find start and end dates that are at intersection of all datasets
for dates in [firedates,smdates]:
    mindate = pd.to_datetime(min(dates))
    maxdate = pd.to_datetime(max(dates))
    if mindate > start_date:
        start_date = mindate
    if maxdate < end_date:
        end_date = maxdate

def date_trim(var, vardates):
    trimmed = var[(pd.to_datetime(vardates)>=start_date) & (pd.to_datetime(vardates)<=end_date),:,:]
    return trimmed

#keep only those dates between the start and end date
fire_trimmed = date_trim(fire, firedates)
sm_trimmed = date_trim(sm, smdates)
sm10_trimmed = date_trim(sm10, smdates)
sm20_trimmed = date_trim(sm20, smdates)
sm30_trimmed = date_trim(sm30, smdates)

#remove 0 values from soil moisture dataset
def no0(dataset):
    dataset[dataset==0] = np.nan
    return dataset
sm_trimmed = no0(sm_trimmed)
sm10_trimmed = no0(sm10_trimmed)
sm20_trimmed = no0(sm20_trimmed)
sm30_trimmed = no0(sm30_trimmed)

#subset dry season only
#dry season defined in Fanin and van der Werf as June-Oct
#June 1 - Oct 29 is 150 days long, from day of year 152 to 302
def dry_subset(dataset_in):
    #start idx = 152 (Jun 1) - 91 (Apr 1) - 1 (python 0 idx) = 60, 426
    #end idx = 302 (Oct 29) - 91 (Apr 1) - 1 (python 0 idx) = 210, 576
    yr2015 = dataset_in[60:210,:,:]
    yr2016 = dataset_in[426:576,:,:]
    out = np.concatenate((yr2015,yr2016), axis=0)
    return out

fire_trimmed = dry_subset(fire_trimmed)
sm_trimmed = dry_subset(sm_trimmed)
sm10_trimmed = dry_subset(sm10_trimmed)
sm20_trimmed = dry_subset(sm20_trimmed)
sm30_trimmed = dry_subset(sm30_trimmed)

#Average every 10 days (axis 0 = days)
def avg10day(var):
    firstdim = int(len(var[:,0,0])/10)
    avg = np.full((firstdim,latsize,lonsize), np.nan)
    for n in np.arange(1,firstdim+1):
        startidx = n*10-10
        endidx = n*10
        avg[n-1,:,:] = np.nanmean(var[startidx:endidx,:,:],axis=0)
    return avg

def sum10day(var,latsize,lonsize):
    firstdim = int(len(var[:,0,0])/10)
    total = np.full((firstdim,latsize,lonsize), np.nan)
    for n in np.arange(1,firstdim+1):
        startidx = n*10-10
        endidx = n*10
        total[n-1,:,:] = np.nansum(var[startidx:endidx,:,:],axis=0)
    return total


#sum the fire, and average the soil moisture
sum_fire = sum10day(fire_trimmed,56,103)
avg_sm = avg10day(sm_trimmed)
avg_sm10 = avg10day(sm10_trimmed)
avg_sm20 = avg10day(sm20_trimmed)
avg_sm30 = avg10day(sm30_trimmed)

#turn nans to 0s
sum_fire[np.isnan(sum_fire)] = 0

#mask ocean pixels for fire
landmask = np.load('G:/My Drive/data/land_ocean/ocean_mask_qtrdeg.npy')
landmask[landmask==0] = np.nan
for i in range(len(sum_fire[:,0,0])):
    sum_fire[i,:,:] = sum_fire[i,:,:] * landmask[331:387,1097:1200]
    

#dates
#start dates
y2015 = np.arange('2015-06-01', '2015-10-29', step=10,dtype='datetime64[D]')
y2016 = np.arange('2016-06-01', '2016-10-29', step=10,dtype='datetime64[D]')
startdates = np.concatenate((y2015,y2016))
enddates = startdates+9

#peat mask
nonpeatarea = np.load('G:/My Drive/data/peat_map/numpy_tables/nonpeatarea_landonly_0.25deg.npy')

##################### PLOTTING
ftsz=18
offset=-0.125
a=0 #SELECT DAY
for tenday in [14+a]:
    
    ###### SETUP PLOT
    fig,(ax1,ax2) = plt.subplots(2,1, sharex=True)
    fig.set_size_inches(10,8)
    plt.subplots_adjust(hspace=.09)
    #Title
    start=str(startdates[tenday])
    end=str(enddates[tenday])
    #fig.suptitle('October 19-28, 2015', fontsize=ftsz)
    #fig.set_dpi(100)
    
    #load lat and lon
    import os
    os.chdir('G:/My Drive/data/gfed_fire/gfed_numpy_tables/')
    fire_lat=np.load('fire_lat.npy')
    fire_lon=np.load('fire_lon.npy')
    x,y=np.meshgrid(fire_lon,fire_lat)

    
    #setup map
    m = Basemap(projection='cyl', llcrnrlat=-6.625, urcrnrlat=7,llcrnrlon=94.5, urcrnrlon=120, resolution='h', lon_0=0, ax=ax1)
    
    
    ###### SOIL MOISTURE ######
    #LOAD DATA
    to_plot = avg_sm10[(tenday),:,:]
    masked_to_plot = np.ma.array(to_plot, mask=np.isnan(to_plot))
    plotter=np.zeros((720,1440))
    plotter2=copy(plotter)
    plotter[331:387,1097:1200] = masked_to_plot
    plotter2[331:387,1097:1200] = nonpeatarea
    
    #PLOT DATA ONTO MAP
    #soil moisture data
    import matplotlib as mpl
    cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["peru","cornflowerblue"])
    cmap.set_bad('white', 1)
    image1 = m.pcolormesh(x+offset,y-offset,np.ma.masked_invalid(plotter), cmap=cmap, vmin=.12, vmax=.22)
    
    #gray peat
    cmap2 = copy(plt.cm.Greys)
    cmap2.set_under('white', 0)
    gray = m.pcolormesh(x+offset,y-offset,plotter2, cmap=cmap2, vmin=1, vmax=2)
    
    # DRAW COASTLINES AND MAP BOUNDARY
    m.drawcoastlines()
    m.drawmapboundary()
    
    # ADD A COLOR BAR
    cb1 = m.colorbar(image1,"right", size="2.5%")
    cb1.set_label('Soil Moisture (cm$^3$/cm$^3$)', fontsize=ftsz)
    cb1.ax.tick_params(labelsize=ftsz)
    
    
    ###### FIRE ######
    m = Basemap(projection='cyl', llcrnrlat=-6.625, urcrnrlat=7,llcrnrlon=94.5, urcrnrlon=120, resolution='h', lon_0=0, ax=ax2)
    
    ## Plot Fire
    #load data
    plotter = np.zeros((720,1440))
    plotter[331:387,1097:1200] = sum_fire[tenday,:,:]
    
    #Plot data onto map
    cmap=mpl.colors.LinearSegmentedColormap.from_list("", ["lightgray","orange","red"])
    cmap.set_under('white', 1)
    image2 = m.pcolormesh(x+offset,y-offset,plotter,shading='flat', cmap=cmap, vmin=0, vmax=.02)
    
    #gray peat
    cmap2 = copy(plt.cm.Greys)
    cmap2.set_under('white', 0)
    gray = m.pcolormesh(x+offset,y-offset,plotter2, cmap=cmap2, vmin=1, vmax=2)
    
    #Add color bar
    cb2 = m.colorbar(image2,"right", size="2.5%", label='Fractional Burned Area')
    cb2.set_label('Fractional Burned Area', fontsize=ftsz)
    from matplotlib import ticker
    tick_locator = ticker.MaxNLocator(nbins=5)
    cb2.locator = tick_locator
    cb2.update_ticks()
    cb2.ax.tick_params(labelsize=ftsz)
    
    # DRAW COASTLINES AND MAP BOUNDARY
    m.drawcoastlines()
    m.drawmapboundary()
    
    fig.text(0.2,.87,'a', fontsize=ftsz, weight='bold')
    fig.text(0.2,.47,'b', fontsize=ftsz, weight='bold')
    
    plt.show()

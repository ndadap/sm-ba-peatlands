
"""
FIGURE 2

Upper panel: Soil moisture timeseries in peatlands colored by fractional burned area, sorted by region. 10 pixels are randomly selected.
Lower panel: Example map of soil moisture values

Contact: Nathan Dadap, ndadap@stanford.edu
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
from mpl_toolkits.basemap import Basemap
import matplotlib as mpl
import matplotlib.gridspec as gridspec


#LOAD DATASETS (Soil Moisture, Fire (burned area fraction), and Peatlands area mask)
sm = np.load('G:/My Drive/data/smap_vod_sm/smap_numpy_tables/MTDCA_0.25deg_10day_avg.npy')
fire = np.load('G:/My Drive/data/gfed_fire/gfed_numpy_tables/fire_large_2015-2016_10day_avg.npy')
peat = np.load('G:/My Drive/data/peat_map/numpy_tables/peatarea_0.25deg.npy')

#GLOBAL SUBSET FOR ISEA
peat = peat[331:387,1097:1200]

#10-DAY DATE VECTOR (MIDPOINT)
datevec = pd.date_range('4/5/2015','12/25/2016', freq='10D')

#SETUP PLOT USING GRID SPEC FOR DIFFERENT PANELS
f=plt.figure(figsize=(8,12),tight_layout=True)
hs=[7,7,2.8,10.5]
ws=[1,1,.03]
axarr=[1,2,3,4,5,6]
gs = gridspec.GridSpec( nrows=4, ncols = 3, height_ratios=hs, width_ratios=ws)
gs.update(left=0.05, right=0.95, bottom=0.08, top=0.93, wspace=0.11, hspace=0.05)
axarr[0]= plt.subplot(gs[0,0])
axarr[1]= plt.subplot(gs[0,1])
axarr[2]= plt.subplot(gs[1,0])
axarr[3]= plt.subplot(gs[1,1])
axarr[4]= plt.subplot(gs[3,:2])

#SET PLOTTING PARAMATERS 
ftsz = 18 #FONT SIZE
sns.set_style("white") #WHITE BACKGROUND
sns.set_style('ticks') #WITH TICKS
number2plot = 10 #NUMBER OF PIXELS TO SHOW 

#ARRAYS TO HOLD LAT AND LON LOCATIONS THAT WILL BE PLOTTED
alllatlocations = np.empty((number2plot*4))
alllonlocations = np.empty((number2plot*4))
allloc_counter = 0

#ITERATE OVER EACH REGION
for reg in np.arange(1,5):
      
    #SELECT LAT/LON TO CYCLE THROUGH BASED ON REGION, AS WELL AS # OF PEATLAND PIXELS
    if reg==1:
        latmult=0
        lonmult=0
        n=26 #KNOWN NUMBER OF PEAT PIXELS
    elif reg==2:
        latmult=0
        lonmult=1
        n=15
    elif reg==3:
        latmult=1
        lonmult=0
        n=40
    else:
        latmult=1
        lonmult=1
        n=47
    
    #CREATE SHUFFLER AS RANDOMIZER FOR NEXT STEP
    arr = np.arange(n)
    np.random.seed(seed=0)
    np.random.shuffle(arr)
    shuffler = arr[:number2plot]

    ### SELECT RANDOM PEAT LOCATIONS TO PLOT
    latlocations = np.empty((number2plot))
    lonlocations = np.empty((number2plot))
    firesize = np.empty((number2plot))
    loc_counter = 0
    iteration = 0
    
    #ITERATE OVER EACH PIXEL IN REGION
    for latidx in range(28):
        for lonidx in range(51):
            latloc=  latidx+28*latmult
            lonloc = lonidx+51*lonmult
            
            #SELECT FROM PEAT LOCATIONS ONLY
            if peat[latloc,lonloc]==0:
                continue       
            
            #KEEP ONLY THOSE DATES WITH VALID SOIL MOISTURE RETRIEVALS
            sm1pix = sm[:,latloc,lonloc]
            smvec = sm1pix[(~np.isnan(sm1pix)) & (sm1pix>0)]
            fire1pix = fire[:,latloc,lonloc]
            firevec = fire1pix[(~np.isnan(sm1pix)) & (sm1pix>0)]
            
            #SELECT ONLY LOCATIONS WITH SUFFICIENT SOIL MOISTURE MEASUREMENTS
            if len(smvec)<50:
                continue               
            
            #ADD RANDOMLY SELECTED LOCATION TO LIST FOR PLOTTING TIMESERIES
            if iteration in shuffler:
                #RECORD LOCATION
                latlocations[loc_counter] = latloc
                lonlocations[loc_counter] = lonloc
                alllatlocations[allloc_counter] = latloc
                alllonlocations[allloc_counter] = lonloc
                firesize[loc_counter] = np.mean(firevec[20:])
                loc_counter+=1
                allloc_counter+=1
            iteration+=1

    #REORDER LOCATIONS BY MEAN FIRE SIZE
    sortedlatlocations = [x for _,x in sorted(zip(firesize,latlocations))][::-1]
    sortedlonlocations = [x for _,x in sorted(zip(firesize,lonlocations))][::-1]
    
    #PLOT REGIONS
    plot=reg-1
    for point in range(number2plot):
    
        #FOR THE SELECTED PIXEL, GET THE LAT AND LON
        latloc = int(sortedlatlocations[point])
        lonloc = int(sortedlonlocations[point])
        
        #KEEP ONLY THOSE DATES WITH VALID SOIL MOISTURE RETRIEVALS
        sm1pix = sm[:,latloc,lonloc]
        fire1pix = fire[:,latloc,lonloc]
        dates1pix = np.arange(64)
        smvec = sm1pix[(~np.isnan(sm1pix)) & (sm1pix>0)]
        firevec = fire1pix[(~np.isnan(sm1pix)) & (sm1pix>0)]
        datesvec = dates1pix[(~np.isnan(sm1pix)) & (sm1pix>0)]
        
        #CHANGE FIRE VALUE TO SMALLEST BURNED AREA FRACTION
        smallest_val=10e-6
        firevec[firevec==0] = smallest_val
        
        # GATHER LINE SEGMENTS AND COLOR ACCORDING TO FIRE
        points = np.array([datesvec, smvec]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        colors = mpl.colors.LinearSegmentedColormap.from_list("", ["lightgray","orange","red"])
        lc = LineCollection(segments, cmap=colors, norm=LogNorm(smallest_val,.2))
        lc.set_array(firevec)
        lc.set_linewidth(1.5)
        
        #PLOT ALL LINE SEGMENTS
        axarr[plot].add_collection(lc)

    
    #PLOT REGION, SET RANGE TO PROVIDE SPACE FOR PLOT
    axarr[plot].set_ylim([0.08,.58])
    axarr[plot].set_xlim([0,60])
    axarr[plot].text(.08,.85,'Region '+str(reg),transform = axarr[plot].transAxes, fontsize=ftsz)
    
    #SET X TICKS
    from matplotlib.ticker import MultipleLocator
    axarr[plot].xaxis.set_major_locator(MultipleLocator(36))
    axarr[plot].xaxis.set_minor_locator(MultipleLocator(12))
    axarr[plot].set_xticklabels(['','Jun 2015','Jun 2016',''], minor=False, fontsize=ftsz-2)
    
    axarr[plot].tick_params(axis='x', which='major',length=7, width=1.5)
    axarr[plot].tick_params(which='minor',length=4, width=1)

#REMOVE X TICKS FOR R1 AND R2 PLOTS, AND Y TICKS FOR R3 AND R4 PLOTS
plt.setp(axarr[0].get_xticklabels(), visible=False)
plt.setp(axarr[1].get_xticklabels(), visible=False)
plt.setp(axarr[1].get_yticklabels(), visible=False)
plt.setp(axarr[3].get_yticklabels(), visible=False)

#SET TICK LABEL SIZE
plt.rc('xtick',labelsize=ftsz)
plt.rc('ytick',labelsize=ftsz)

#COLORBARS
f.subplots_adjust(right=0.8)
height=.7
cbar_ax = plt.subplot(gs[:2,2])
cbar=mpl.colorbar.ColorbarBase(ax=cbar_ax, cmap=colors, norm=LogNorm(smallest_val,.2))
cbar.set_label('Fractional Burned Area', fontsize=ftsz)

#AXIS LABELS
f.text(0.48, 0.435, 'Date', ha='center', va='center', fontsize=ftsz+2)
f.text(-.05, 0.71, 'Soil moisture (cm$^3$/cm$^3$)', ha='center', va='center', rotation='vertical', fontsize=ftsz+2)

#### PLOT SAMPLE LOCATIONS
#LOAD LAT AND LON
fire_lat=np.load('G:/My Drive/data/gfed_fire/gfed_numpy_tables/fire_lat.npy') #LATITUDE VECTOR 0.25 DEGREE INCREMENTS
fire_lon=np.load('G:/My Drive/data/gfed_fire/gfed_numpy_tables/fire_lon.npy') #LONGITUDE VECTOR
lon,lat=np.meshgrid(fire_lon,fire_lat)

#PLOTTING
m = Basemap(projection='cyl', llcrnrlat=-6.875, urcrnrlat=7, llcrnrlon=94.5, urcrnrlon=120, resolution='h', lon_0=0, ax=axarr[4], suppress_ticks=False)

#SOIL MOISTURE EXAMPLE
pmsk = np.load('G:/My Drive/data/peat_map/numpy_tables/peatarea_0.25deg.npy') #// 0.25 DEG PEAT AREA MASK
plotter = np.zeros((720,1440))
period=20
plotter[331:387,1097:1200] =  pmsk[331:387,1097:1200] * sm[period,:,:]
cmap = mpl.cm.Blues
cmap.set_under('white', 1.)
image = m.pcolormesh(lon-.125,lat+.125,plotter,shading='flat',cmap=cmap,vmin=0.05, vmax=0.35)

#DRAW ARROW POINTING TO AXIS
for i in [2,3]:
    axarr[i].annotate(s='Snapshot\nbelow', xy=(period/len(datevec),0), xytext=(period/len(datevec)-.125,-.3),  xycoords='axes fraction', arrowprops=dict(facecolor='red', shrink=0.05), annotation_clip=False, fontsize=ftsz-4)

#COLORBAR
from matplotlib.colorbar import Colorbar
cbar_ax2 = plt.subplot(gs[-1,-1])
cbar2 = Colorbar(ax=cbar_ax2, mappable=image, orientation="vertical")
cbar2.set_label('Soil Moisture (cm$^3$/cm$^3$)', fontsize=ftsz)

#POINTS
for idx in range(len(alllatlocations)):
    lat_in = lat[int(alllatlocations[idx]+331),0]
    lon_in = lon[0,int(alllonlocations[idx]+1097)]
    dot = m.plot(lon_in, lat_in, markeredgecolor='r', markerfacecolor='lime', markeredgewidth=0.75, marker='D', markersize=3)
    offset=(-15,-15)

axarr[4].axvline((94.5+120)/2, color='grey', linestyle='--', linewidth = 1.5)
axarr[4].axhline((-6.875+7)/2, color='grey', linestyle='--', linewidth = 1.5)

#ALTERNATE LABELING OF REGIONS
cl = 'black'
plt.text(.0085, .919,'R1',  transform = axarr[4].transAxes, fontsize=ftsz,color=cl)
plt.text(.5085, .919,'R2',  transform = axarr[4].transAxes, fontsize=ftsz,color=cl)
plt.text(.0085, .419,'R3',  transform = axarr[4].transAxes, fontsize=ftsz,color=cl)
plt.text(.5085, .419,'R4',  transform = axarr[4].transAxes, fontsize=ftsz, color=cl)

#CUSTOM LEGEND
axarr[4].legend(handles= [mpl.lines.Line2D([], [], markeredgecolor='r', markerfacecolor='lime', markeredgewidth=0.75, marker='D', linestyle='None', markersize=3, label='Sample pixel')], loc=[-.02,0], prop={'size':ftsz}, handletextpad=-.5)

# DRAW COASTLINES AND MAP BOUNDARY
m.drawcoastlines()
m.drawmapboundary()
lonticks = [107.25-8.25,107.25,107.25+8.25]
latticks = [-4,0,4]
axarr[4].set_xticks(lonticks)
axarr[4].set_yticks(latticks)
axarr[4].set_xticklabels(['99$^{\circ}$E',str(lonticks[1])+'$^{\circ}$E',str(lonticks[2])+'$^{\circ}$E'])
axarr[4].set_yticklabels(['4$^{\circ}$S','0$^{\circ}$','4$^{\circ}$N'])

#PLOT LABEL LETTER
f.text(-.03,.925,'a', fontsize=ftsz, weight='bold')
f.text(-.03,.39,'b', fontsize=ftsz, weight='bold')

plt.show()



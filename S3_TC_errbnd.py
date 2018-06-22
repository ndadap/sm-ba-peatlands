"""
Purpose: Calculate TC upperbound

Contact: Nathan Dadap, ndadap@stanford.edu
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

#define covariance function using numpy function (to shorten)
def cov(v1,v2):
    output = np.cov(v1,v2)[0][1]
    return output

##### LOAD DATASETS
dataset = 'MTDCA'

#SMAP MTDCA
s_data = np.load('G:/My Drive/data/smap_vod_sm/smap_numpy_tables/201504-201703_smap_mtdca_daily_masked.npy')
unique_mask = np.load('G:/My Drive/data/smap_vod_sm/smap_numpy_tables/smap_mtdca_unique_mask.npy') 

#GLDAS:
g_data = np.load('G:/My Drive/data/gldas/downscaled_gldas.npy')

##### PERFORM TRIPLE COLLOCATION TO CALCULATE UBRMSE

#set lag (days)
lag = 1

#create empty matrix to fill with error variance info
err_var_s = np.zeros((198,276)) #SMAP error variance
err_var_g = np.zeros((198,276)) #GLDAS error variance
err_var_glag = np.zeros((198,276)) #GLDAS lag error variance for point of comparison

err_var_s_dif = np.zeros((198,276)) #alternate difference method
err_var_g_dif = np.zeros((198,276)) # alternate difference method
g_beta = np.zeros((198,276)) #rescaling parameter
g_lag_beta = np.zeros((198,276)) #rescaling parameter for g_lag

#other information of interest
num_samples = np.zeros((198,276)) #number of samples per pixel
corr_coef = np.zeros((198,276)) #correlation coefficent
s_moving_avg = np.zeros((731,198,276)) #smap moving average
g_auto_corr_gaps = np.zeros((198,276)) #gldas auto correlation
g_auto_corr = np.zeros((198,276)) #gldas auto correlation

########
testpixlatidxs = [20,174,111,130,141,75]
testpixlonidxs = [80,113,83,183,214,215]

testpixlatidxs = [95,123,145,139,132]
testpixlonidxs = [85,101,117,177,210]

#hold error eq parameters
covsg=np.empty(len(testpixlatidxs))
covsgl=np.empty(len(testpixlatidxs))
covggl=np.empty(len(testpixlatidxs))
varg=np.empty(len(testpixlatidxs))

err_vals=np.empty(len(testpixlatidxs))
########


#loop over each pixel spatially
for idx in range(len(testpixlatidxs)):
    lat=testpixlatidxs[idx]
    lon=testpixlonidxs[idx]
   
    #get 1d data (time) for pixel. create lag vector
    s_raw_1d =      s_data[:,lat,lon]
    g_long =        g_data[:,lat,lon]
    g_lag_long =    np.roll(g_data[:,lat,lon], lag) 
    
    #skip iteration if there are less than n-real values for pixel
    n_real = 50
    if np.count_nonzero(~np.isnan(s_raw_1d)) < n_real:
        err_var_s[lat,lon]=np.NaN
        err_var_g[lat,lon]=np.NaN
        
        err_var_s_dif[lat,lon]=np.NaN
        err_var_g_dif[lat,lon]=np.NaN
        g_beta[lat,lon]=np.nan
        
        g_auto_corr_gaps[lat,lon] = np.nan
        g_auto_corr[lat,lon] = np.nan
        
        g_beta[lat,lon]=np.nan
        g_lag_beta[lat,lon] = np.nan
        continue

    #keep only days where there is a smap value
    s =         s_raw_1d[np.isnan(s_raw_1d)==False]
    g =         g_long[np.isnan(s_raw_1d)==False] # create separate vector for subset
    g_lag =     g_lag_long[np.isnan(s_raw_1d)==False] #create separate vector for subset
    ########

    try:
        covsg[idx] = cov(s,g)
        covsgl[idx] = cov(s,g_lag)
        covggl[idx] = cov(g_long,g_lag_long)
        varg[idx] = np.var(g)
        
    except ValueError:
        print('missed')
        continue
    
    ########
    
    ### calculate error variance from s, g, and g_lag
    err_vals[idx]= np.var(s) - cov(s,g)*cov(s,g_lag)/cov(g,g_lag)
    err_var_g[lat,lon] = np.var(g) - cov(g,g_lag)*cov(g,s)/cov(s,g_lag)
    err_var_glag[lat,lon] = np.var(g_lag) - cov(g_lag,s)*cov(g_lag,g)/cov(g,s)
    
    ### calculate rescaling parameter of g with respect to s (see Gruber paper A.4)
    g_beta[lat,lon] = cov(s,g_lag)/cov(g,g_lag)
    g_lag_beta[lat,lon] = cov(s,g)/cov(g_lag,g)
    
    ### calculate error variance using alternate difference method
    # first calculate g rescaled to s (see Gruber paper eq A.2)
    g_s = g_beta[lat,lon]*(g-np.nanmean(g)) + np.nanmean(s)
    g_lag_s = g_lag_beta[lat,lon]*(g_lag - np.nanmean(g_lag)) + np.nanmean(s)
    # next calculate differences (Gruber eq A.7)
    err_var_s_dif[lat,lon] = np.mean(np.multiply((s-g_s),(s-g_lag_s)))
    err_var_g_dif[lat,lon] = np.mean(np.multiply((g_s-s),(g_s-g_lag_s))) / g_beta[lat,lon]**2
    
    ### calculate correlation coefficient
    corr_coef[lat,lon] = np.corrcoef(s,g)[0,1]
    
    ### gldas autocorrelation
    g_auto_corr_gaps[lat,lon] = np.corrcoef(g,g_lag)[0,1]
        
########## PLOTTING
#SETUP PLOTS
size=2
from matplotlib import gridspec
fig=plt.figure(figsize=(5*size,1.5*size))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.5])
gs.update(wspace=0.1)
ax1=plt.subplot(gs[0])
ax2=plt.subplot(gs[1])
ftsz=16

### ubRMSE AS A FUNCTION OF AUTOCORRELATION
rho_vals = np.linspace(0,1,100)
for idx in range(len(testpixlatidxs)):
    #equations describing distance from bound to true:
    var_eg = (covsgl[idx]*varg[idx] - covggl[idx]*covsg[idx]) / (covsgl[idx] - rho_vals*covsg[idx])
    E = rho_vals*var_eg
    
    dT = covsg[idx]*covsgl[idx]*E/(covggl[idx]**2-covggl[idx]*E) #change in error
    error=err_vals[idx]**.5-dT[dT>0]**0.5 #upperbound minus change
    
    #plotting
    ax1.plot(rho_vals[dT>0],error,label=str(idx+1),linewidth=3) #need absolute val for negative values
    ax1.scatter(0,err_vals[idx]**0.5, marker='*', s=105)
    
    ax1.set_ylabel('ubRMSE (cm$^3$/cm$^3$)', fontsize=ftsz)
    ax1.set_xlabel('GLDAS error autocorrelation', fontsize=ftsz)
    ax1.set_ylim((0,.1))

ax1.legend(fontsize=ftsz-3)
ax1.xaxis.set_tick_params(labelsize=ftsz)
ax1.yaxis.set_tick_params(labelsize=ftsz)

### TEST PIXELS LOCATIONS
import numpy as np
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

#load ease lat and lon
f = scipy.io.loadmat('G:/My Drive/data/smap_vod_sm/latlonAP.mat')
lat=np.array(f.get('lat'))
lon=np.array(f.get('lon'))
x,y=np.meshgrid(lon[1,:],lat[:,1])

#PLOTTING
m = Basemap(projection='cyl', llcrnrlat=-6.875, urcrnrlat=7,
            llcrnrlon=94.5, urcrnrlon=120, resolution='h', lon_0=0, ax=ax2)

pmsk = np.load('G:/My Drive/data/peat_map/numpy_tables/peatarea_9km.npy')
cmap = matplotlib.cm.ocean_r
cmap.set_under('white', 1.)
image = m.pcolormesh(x,y,pmsk,shading='flat',cmap=cmap,vmin=0, vmax=12)

for idx in range(len(testpixlatidxs)):
    lat_in = lat[testpixlatidxs[idx]+711,0]
    lon_in = lon[0,testpixlonidxs[idx]+2940]
    m.plot(lon_in, lat_in, 'r.',markersize=10)
    offset=(-15,-15)
    if idx>2:offset=(5,5)
    plt.annotate(str(idx+1),xy=(lon_in,lat_in),xytext=offset,textcoords='offset points', fontsize=ftsz)

ax2.set_xlabel('Locations', fontsize=ftsz)

# DRAW COASTLINES AND MAP BOUNDARY
m.drawcoastlines()
m.drawmapboundary()


ax1.text(-.34,.96,'a', weight='bold', fontsize=ftsz,transform=ax1.transAxes)
ax2.text(-.05,.96,'b', weight='bold',fontsize=ftsz, transform=ax2.transAxes)

plt.show()
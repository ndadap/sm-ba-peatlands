"""
Purpose: Calculate ubRMSE using Triple Collocation

Contact: Nathan Dadap, ndadap@stanford.edu
"""

import scipy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#COVARIANCE FUNCTION
def cov(v1,v2):
    output = np.cov(v1,v2)[0][1]
    return output


#LOAD MASK TO ENSURE CALCULATIONS OVER SAME SUBSET OF PIXELS
all_mask = np.load('G:/My Drive/scripts/SMAPL2processing/numpy_tables/all_mask.npy')

#LOAD SMAP DATA
s_data = np.load('G:/My Drive/data/smap_vod_sm/smap_numpy_tables/201504-201703_smap_mtdca_daily_masked.npy')

#LOAD GLDAS DATA
g_data = np.load('G:/My Drive/data/gldas/downscaled_gldas.npy')


##### PERFORM TRIPLE COLLOCATION

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


#loop over each pixel spatially
for lat in range(len(s_data[0,:,0])):
    for lon in range(len(s_data[0,0,:])):
       
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

        ### calculate error variance from s, g, and g_lag
        err_var_s[lat,lon] = np.var(s) - cov(s,g)*cov(s,g_lag)/cov(g_long,g_lag_long)
        err_var_g[lat,lon] = np.var(g_long) - cov(g_long,g_lag_long)*cov(g,s)/cov(s,g_lag)
        err_var_glag[lat,lon] = np.var(g_lag_long) - cov(g_lag,s)*cov(g_lag_long,g_long)/cov(g,s)
        
        ### calculate rescaling parameter of g with respect to s (see Gruber paper A.4)
        g_beta[lat,lon] = cov(s,g_lag)/cov(g_long,g_lag_long)
        g_lag_beta[lat,lon] = cov(s,g)/cov(g_lag_long,g_long)
        
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
        

err_var_s[err_var_s<=0]=np.nan

#calculate avg error in peatlands
peat_mask=np.load('G:/My Drive/data/peat_map/numpy_tables/peatarea_9km.npy')
non_peat_mask = peat_mask[711:909,2940:3216]
peat_mask[peat_mask==0]=np.nan
p_only_err = peat_mask[711:909,2940:3216]*err_var_s**0.5
peat_avg_err = np.nanmean(p_only_err)
p_err = round(peat_avg_err,3)

#calculate avg error in non-peatlands
non_peat_mask[non_peat_mask==1]=0
non_peat_mask[np.isnan(non_peat_mask)]=1
non_peat_mask[non_peat_mask==0]=np.nan
non_p_err = non_peat_mask*err_var_s**0.5
non_p_avg_err = round(np.nanmean(non_p_err),3)


##### PLOT
from mpl_toolkits.basemap import Basemap

#LAT AND LON VECTORS
f = scipy.io.loadmat('G:/My Drive/data/smap_vod_sm/latlonAP.mat')
lat=np.array(f.get('lat'))
lon=np.array(f.get('lon'))
x,y=np.meshgrid(lon[1,:],lat[:,1])

#LOAD DATA
peat_area=np.zeros((1624,3856))
peat_area[711:909,2940:3216] = np.ma.array(err_var_s**0.5, mask = np.isnan(err_var_s))


#SETUP PLOT
fig,ax = plt.subplots(figsize=(13,7))
#fig.dpi=300
res='h'
sns.set(font_scale = 2)
sns.set_style("white", {'xtick.major.size': 3.0,'ytick.major.size': 3.0})

#INSET FIGURE
fig.subplots_adjust(left=0., right=1., bottom=0., top=.9)
m = Basemap(projection='cyl', llcrnrlat=-6.875, urcrnrlat=7,
            llcrnrlon=93.5, urcrnrlon=120, resolution=res, lon_0=0)


# PLOT DATA ONTO MAP
from copy import copy
palette = copy(plt.cm.viridis_r)
palette.set_over('white', 1.0)
palette.set_under('white', 1.0)
palette.set_bad('b', 1.0)
palette2 = copy(plt.cm.binary)
palette2.set_over(alpha=0.0)
palette2.set_under(alpha=0.0)
palette2.set_bad('b', 1.0)

image = m.pcolormesh(x,y,peat_area,shading='flat',cmap=palette,vmin=0.04, vmax=0.12)


# DRAW COASTLINES AND MAP BOUNDARY
m.drawcoastlines(ax=ax)
m.drawmapboundary(ax=ax)

#FONT SIZE
ftsz=26

# ADD A COLOR BAR
cb = m.colorbar(image,"right", size="1%", pad=.05, ax=ax)#, label='soil moisture')
cb.ax.tick_params(labelsize=ftsz)
cb.ax.set_ylabel('Estimated ubRMSE (cm$^3$/cm$^3$)',fontsize=ftsz)

#PLOT ERR DIST IN SUBSET
left, bottom, width, height = [0.11, 0.15, 0.16, 0.2]
ax2 = fig.add_axes([left, bottom, width, height])
sns.distplot(p_only_err[~np.isnan(p_only_err)], bins=20, hist=False, kde=True,ax=ax2,kde_kws={'clip': (0.03, .16)})
ax2.set_xlabel('ubRMSE (cm$^3$/cm$^3$)', fontsize=ftsz-4)
ax2.set_ylabel('Probability density', fontsize=ftsz-4)
ax2.set_xticks([0.05,0.1])
ax2.set_ylim((0,30))
ax2.annotate('mean= \n'+str(p_err), xy=(.55,.6), xycoords='axes fraction', fontsize=ftsz-4)
sns.despine(ax=ax2, offset=5)

plt.show()
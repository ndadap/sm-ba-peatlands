
"""
FIGURE 1. CDF and PDF of vegetation transmissivity in peat vs non-peat areas.

Contact: Nathan Dadap, ndadap@stanford.edu
"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import io



#LOAD DATA // data is a 3856 x 1624 pixel array with per pixel annual mean VOD
vod_mean = np.load('G:/My Drive/data/smap_vod_sm/smap_numpy_tables/ease_vod_annual_mean.npy')

#SETUP GRID // data is EASE-2 grid lat and lon
f = io.loadmat('G:/My Drive/data/smap_vod_sm/latlonAP.mat')
datalat=np.array(f.get('lat'))
datalon=np.array(f.get('lon'))

#CALCULATE ATTENUATION
atten_mat = np.transpose(np.ma.masked_invalid(np.exp(-vod_mean/np.cos(0.698132))))

#TAKE VALUES IN PEAT LOCATIONS // array of peatland pixels (non-peat areas = 0, peat areas = 1)
peat_mask=np.load('G:/My Drive/data/peat_map/numpy_tables/peatarea_9km.npy')

#TAKE VALUES IN NON-PEAT LOCATIONS
non_peat_mask = np.empty((1624,3856))
non_peat_mask[peat_mask==0] = 1
non_peat_mask[peat_mask==1] = 0

#MULTIPLYING TO GET VALUES
peat_values=np.multiply(atten_mat,peat_mask)
non_peat_values=np.multiply(atten_mat,non_peat_mask)

#ONLY THIS REGION, FLATTEN, AND REMOVE NAN FOR HISTOGRAM
final_peat=np.ndarray.flatten(np.nan_to_num(peat_values[711:909,2940:3216])) #index for regional area
final_non_peat=np.ndarray.flatten(np.nan_to_num(non_peat_values[711:909,2940:3216]))



########### PROBABILITY DISTRIBUTION FUNCTION
ftsz=26 #FONTSIZE
lnwdth = 4 #LINEWIDTH
fig,ax = plt.subplots(figsize=(8.5,7))
sns.set(font_scale = 2)
sns.set_style("white", {'xtick.major.size': 3.0,'ytick.major.size': 3.0})

#PREPARE HISTOGRAM POINTS
hist1, bins = np.histogram(final_non_peat, bins=30, normed=True, range=[0.1,0.8])
hist2, bins = np.histogram(final_peat, bins=30, normed=True, range=[0.1,0.8])
bin_centers = (bins[1:]+bins[:-1])*0.5

#PLOT HISTOGRAM WITH LEGEND
line1 = plt.plot(bin_centers, hist1, 'r', ls='dashed', linewidth = lnwdth)
line2 = plt.plot(bin_centers, hist2, 'b', linewidth =lnwdth)
line3 = plt.axvline(x=np.exp(-0.766/np.cos(0.698132)), linestyle='-.', color='grey', linewidth = lnwdth-1) #VERTICAL LINE FOR PENETRATION THRESHOLD

#PLOT VISUAL PARAMETERS
plt.xlim(.1,.8)
plt.xticks(np.arange(.1,.9,.1))
plt.ylim(0,5)
sns.despine(offset=14)
plt.tick_params(axis='both', which='major', labelsize=ftsz, length=7, direction='in')
plt.legend(['Non-peat', 'Peat', 'Threshold'], fontsize=ftsz, frameon=False, bbox_to_anchor=(.55,1))

#AXIS LABELS
plt.xlabel('Vegetation Transmissivity', fontsize=ftsz)
plt.ylabel('Probability Density', fontsize=ftsz)

plt.show()



########### CUMULATIVE DISTRIBUTION FUNCTION
fig,ax = plt.subplots(figsize=(8.5,7))

#SET DATA TO BINARY
peat = final_peat[final_peat>0]
nonpeat= final_non_peat[final_non_peat>0]

#PLOT DATA
plt.plot(np.sort(nonpeat), np.linspace(0, 1, len(nonpeat), endpoint=False),color='r', linewidth = lnwdth, linestyle='-.')
plt.plot(np.sort(peat), np.linspace(0, 1, len(peat), endpoint=False), color='b', linewidth = lnwdth)

#PLOT THRESHOLD LINE
plt.axvline(x=np.exp(-0.766/np.cos(0.698132)), linestyle='--', color='grey', linewidth = lnwdth-2)

#PLOT VISUAL PARAMETERS
plt.xlim(.1,.9)
plt.xticks(np.arange(.1,1,.1))
plt.ylim(0,1)
sns.despine(offset=14)
plt.tick_params(axis='both', which='major', labelsize=ftsz, length=7, direction='in')

#LABELS
plt.legend(['Non-peat', 'Peat', '1/e Penetration'], fontsize=ftsz, frameon=False)
plt.xlabel('Vegetation Transmissivity', fontsize=ftsz)
plt.ylabel('Fraction of Data', fontsize=ftsz)

plt.show()

"""
FIGURE 3

Left panel: Burned area, binned by soil moisture data at varying lead times.
Right panel: Burned area - soil moisture plot (Blue). Frandsen ignition probability curve for peat. (Red)

Contact: Nathan Dadap, ndadap@stanford.edu
"""

from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#LOAD VARIABLES: FIRE AND SOIL MOISTURE (VARYING LEAD TIMES)
#ARRAY AXES 0, 1, AND 2 CORRESPOND TO TIME, LATITUDE AND LONGITUDE
fire = np.load('G:/My Drive/data/gfed_fire/gfed_numpy_tables/fire_small_2015-2016_daily.npy')
sm = np.load('G:/My Drive/data/smap_vod_sm/smap_numpy_tables/MTDCA_0.25deg_daily.npy')

#GET MAX SOIL MOISTURE VALUE
max_sm = np.nanmax(sm)

#CREATE DATE RANGE ASSOCIATED WITH EACH DATASET
firedates = pd.date_range('1/1/2015','12/31/2016', freq='D')
smdates = pd.date_range('4/1/2015','3/31/2017', freq='D')


########## DATASET PREPROCESSING

#CREATE COPIES OF SOIL MOISTURE DATA AT VARYING LEAD TIMES
for lead in [10,20,30]:
    
    #MAKE COPY OF SOIL MOISTURE ARRAY, SUBSET, AND FILL FRONT OF TIMESERIES WITH NAN
    #2015-2016 COMPRISES 731 DAYS
    lag_var = np.concatenate((np.full((lead,56,103),np.nan),sm[:(731-lead),:,:]),axis=0)
    lag_var[:lead] = np.nan
    if lead == 10:
        sm10 = lag_var
    if lead == 20:
        sm20 = lag_var
    if lead == 30:
        sm30 = lag_var

#INITIALIZE GENERAL START AND END TIME FOR DATA
start_date, end_date = pd.to_datetime('2014-01-01'), pd.to_datetime('2018-01-01')

#SELECT ONLY DATES THAT ARE AT THE INTERSECTION OF BOTH DATASETS
for dates in [firedates,smdates]:
    mindate = pd.to_datetime(min(dates))
    maxdate = pd.to_datetime(max(dates))
    if mindate > start_date:
        start_date = mindate
    if maxdate < end_date:
        end_date = maxdate

#TRIM DATA TO BE ONLY FOR DATES BETWEEN START AND END OF OVERLAPPING DATES
def date_trim(var, vardates):
    trimmed = var[(pd.to_datetime(vardates)>=start_date) & (pd.to_datetime(vardates)<=end_date),:,:]
    return trimmed

fire_trimmed = date_trim(fire, firedates)
sm_trimmed = date_trim(sm, smdates)
sm10_trimmed = date_trim(sm10, smdates)
sm20_trimmed = date_trim(sm20, smdates)
sm30_trimmed = date_trim(sm30, smdates)

#SET ERRONEOUS SOIL MOISTURE RETRIEVALS (VALUE 0) TO NAN
def no0(dataset):
    dataset[dataset==0] = np.nan
    return dataset

sm_trimmed = no0(sm_trimmed)
sm10_trimmed = no0(sm10_trimmed)
sm20_trimmed = no0(sm20_trimmed)
sm30_trimmed = no0(sm30_trimmed)


#TAKE ONLY DATA FROM DRY SEASON
# // dry season in this region is defined as June-Oct (See Fanin et al. 2017)
# // June 1 - Oct 29 is 150 days long, from day of year 152 to 302
def dry_subset(dataset_in):
    yr2015 = dataset_in[60:210,:,:] #select day of 
    yr2016 = dataset_in[426:576,:,:]
    out = np.concatenate((yr2015,yr2016), axis=0)
    return out

fire_trimmed = dry_subset(fire_trimmed)
sm_trimmed = dry_subset(sm_trimmed)
sm10_trimmed = dry_subset(sm10_trimmed)
sm20_trimmed = dry_subset(sm20_trimmed)
sm30_trimmed = dry_subset(sm30_trimmed)

#AVERAGE SOIL MOISTURE AND SUM BURNED AREA OVER 10-DAY WINDOWS
def avg10day(var):
    firstdim = int(len(var[:,0,0])/10)
    avg = np.full((firstdim,56,103), np.nan)
    for n in np.arange(1,firstdim+1):
        startidx = n*10-10
        endidx = n*10
        avg[n-1,:,:] = np.nanmean(var[startidx:endidx,:,:],axis=0)
    return avg
def sum10day(var):
    firstdim = int(len(var[:,0,0])/10)
    total = np.full((firstdim,56,103), np.nan)
    for n in np.arange(1,firstdim+1):
        startidx = n*10-10
        endidx = n*10
        total[n-1,:,:] = np.nansum(var[startidx:endidx,:,:],axis=0)
    return total
sum_fire = sum10day(fire_trimmed)
avg_sm = avg10day(sm_trimmed)
avg_sm10 = avg10day(sm10_trimmed)
avg_sm20 = avg10day(sm20_trimmed)
avg_sm30 = avg10day(sm30_trimmed)

#TAKE SUBSET OF DATA IN PEATLANDS ONLY
peatmask = np.load('G:/My Drive/data/peat_map/numpy_tables/peatarea_0.25deg.npy') #// peat=1, nonpeat=nan
peatmask = peatmask[331:387,1097:1200]
def apply_peatmask(datain):
    for day in range(len(datain[:,0,0])):
        datain[day,:,:] = datain[day,:,:] * peatmask
    return datain
sum_fire = apply_peatmask(sum_fire)
avg_sm = apply_peatmask(avg_sm)
avg_sm10 = apply_peatmask(avg_sm10)
avg_sm20 = apply_peatmask(avg_sm20)
avg_sm30 = apply_peatmask(avg_sm30)

#FLATTEN DATA FOR PLOTTING
f = np.ndarray.flatten(sum_fire)
sm = np.ndarray.flatten(avg_sm)
sm10 = np.ndarray.flatten(avg_sm10)
sm20 = np.ndarray.flatten(avg_sm20)
sm30 = np.ndarray.flatten(avg_sm30)




############# PLOTTING RESULTS
#SET FONTSIZE
ftsz = 26

#PLOT FORMAT PARAMETERS
sns.set_style("white")
sns.set_style("white", {'xtick.major.size': 3.0,'ytick.major.size': 3.0}) #tick size
sns.set_style({"xtick.direction": "in","ytick.direction": "in"}) #tick direction
plt.rc('xtick',labelsize=ftsz)
plt.rc('ytick',labelsize=ftsz)

### CREATE SOIL MOISTURE BINS BY EQUAL POPULATION SIZE
#SORT VALUES
realval_sm = np.sort(sm[~np.isnan(sm)])
#SET NUMBER OF BINS
num_bins = 15
#CREATE VECTOR TO HOLD INTERVAL VALUES
intervals = np.full((num_bins),np.nan)
for i in range(num_bins):
    intervals[i] = realval_sm[int(len(realval_sm)*i/num_bins)]
#APPEND MIN SM VALUE FOR FIRST BIN
intervals = np.append(0, intervals)
#APPEND MAX SM VALUE FOR LAST BIN
intervals = np.append(intervals, max_sm)

#ITERATE OVER VARIOUS SOIL MOISTURE LEAD TIMES
for idx,var in enumerate([sm,sm10,sm20,sm30]):
    
    #USE HISTOGRAM WITH NEARLY EQUAL SIZED BINS
    #BIN PIXELS BY SOIL MOISTURE, THEN WEIGHT BINS WITH BURNED AREA PER PIXEL
    weighted = np.histogram(var, bins=intervals, weights=f, density=False, range=[min(intervals),max(intervals)])[0]
    bin_count = np.histogram(var, bins=intervals, density=False, range=[min(intervals),max(intervals)])[0]
    height = weighted / bin_count #HEIGHT = MEAN BURNED AREA WITHIN A GIVEN SOIL MOISTURE BIN
    bins = intervals            
    bin_centers = (bins[1:]+bins[:-1])*0.5
    
    #GET STANDARD DEVIATION AND BIN COUNT 
    stdvs = np.full((len(bin_centers)),np.nan)
    count = np.full((len(bin_centers)),np.nan)
    sterr = np.full((len(bin_centers)),np.nan)
    height2 = np.full((len(bin_centers)),np.nan) #for comparison
    for num,x in enumerate(bin_centers):
        #CREATE MASK TO GET VALUES FROM ONE BIN AT A TIME
        bin_mask = [(var > bins[num]) & (var < bins[num+1])]
        var_in_bin = var[bin_mask]
        fire_in_bin = f[bin_mask]
        
        #GET STANDARD DEVIATION AND COUNT TO CALCULATE STANDARD ERROR     
        stdvs[num] = np.nanstd(fire_in_bin)
        def rlct(vector): return np.count_nonzero(~np.isnan(vector)) #count num of real values
        count[num] = rlct(var_in_bin*fire_in_bin)
        sterr[num] = np.nanstd(fire_in_bin)/(rlct(var_in_bin*fire_in_bin))**0.5
        
    #LINESTYLE AND LABEL (VECTOR)
    line = ["-","--","-.",":"]
    lab = ['Soil moisture\n(no lead)', '10-day lead', '20-day lead', '30-day lead']
    
    #CHOOSE SUBPLOT
    if idx==0:
        fig, axarr = plt.subplots(1,2,figsize=(13,6), tight_layout=True)
    
    #PLOT VALUES
    axarr[0].plot(bin_centers,height, label=lab[idx], linewidth=3, ls=line[idx])
    axarr[0].fill_between(bin_centers, height+sterr, height-sterr, alpha=0.5)

#SET LABELS, PARAMETERS, LEGEND
axarr[0].set_xlabel("Soil moisture (cm$^3$/cm$^3$)")
axarr[0].set_ylabel("Fractional burned area")
axarr[0].set_xticks(ticks=np.arange(0.1,0.6,0.1)) #tick locations and interval
axarr[0].set_yticks(ticks=[0,5e-3,10e-3,15e-3,20e-3])
axarr[0].set_ylim((0,.02))
axarr[0].set_xlim((0.1,.5))
axarr[0].tick_params(length=7,width=2)
axarr[0].legend(fontsize=ftsz-4, frameon=False, edgecolor='black')#, framealpha=1)



########### REPEAT PLOTTING FROM LEFT PANEL FOR CONCURRENT (0-LEAD) SOIL MOISTURE

#PLOT VISUALIZATION PARAMETERS
sns.set_style("white")
sns.set_style("white", {'xtick.major.size': 3.0, 'ytick.major.size': 3.0}) #tick size
sns.set_style({"xtick.direction": "in","ytick.direction": "in"}) #tick direction
plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)

#ESTABLISH BINS
realval_sm = np.sort(sm[~np.isnan(sm)])
num_bins = 15
intervals = np.full((num_bins),np.nan)
for i in range(num_bins):
    intervals[i] = realval_sm[int(len(realval_sm)*i/num_bins)]
intervals = np.append(intervals, 0.57661124700231015)
intervals = np.append(0, intervals)

## SEE COMMENTED CODE ABOVE
for idx,var in enumerate([sm]): #PLOT ONLY SOIL MOISTURE WITH 0-LEAD
    
    weighted = np.histogram(var,bins=intervals,weights=f,density=False, range=[min(intervals),max(intervals)])[0]
    bin_count = np.histogram(var,bins=intervals,density=False,range=[min(intervals),max(intervals)])[0]
    height = weighted / bin_count #this gives us the mean
    bins = intervals            
    bin_centers = (bins[1:]+bins[:-1])*0.5
    
    stdvs = np.full((len(bin_centers)),np.nan)
    count = np.full((len(bin_centers)),np.nan)
    sterr = np.full((len(bin_centers)),np.nan)
    height2 = np.full((len(bin_centers)),np.nan) #for comparison
    for num,x in enumerate(bin_centers):
        bin_mask = [(var > bins[num]) & (var < bins[num+1])]
        var_in_bin = var[bin_mask]
        fire_in_bin = f[bin_mask]
        stdvs[num] = np.nanstd(fire_in_bin)
        def rlct(vector): return np.count_nonzero(~np.isnan(vector))
        count[num] = rlct(var_in_bin*fire_in_bin)
        sterr[num] = np.nanstd(fire_in_bin)/(rlct(var_in_bin*fire_in_bin))**0.5
    line = ["-","--","-.",":"]
    lab = ['Soil moisture\n (no lead)', '10-day lead', '20-day lead', '30-day lead']
    axarr[1].plot(bin_centers,height, label='10-day lead', linewidth=3, ls=line[idx], color='b')
    axarr[1].fill_between(bin_centers, height+sterr, height-sterr, alpha=0.5)

axarr[1].set_xlabel("Soil moisture (cm$^3$/cm$^3$)")
axarr[1].set_ylabel("Fractional burned area", color='b')
axarr[1].set_xticks(ticks=np.arange(0.1,0.6,0.1)) #tick locations and interval
axarr[1].set_yticks(ticks=np.arange(0,0.025,0.005))
axarr[1].tick_params('y',colors='blue')
axarr[1].set_ylim((0,.02))
axarr[1].set_xlim((0.1,.5))
axarr[1].tick_params(length=7,width=2)


### PLOT FRANDSEN IGNITION PROBABILITY MODEL

#PEAT PARAMETERS
#INORGANIC CONTENT PERCENTAGE ()
ioc = 9.4
ioc_cv = .071 #COEFFICIENT OF VARIATION
#ORGANIC BULK DENSITY (kg/m^3)
obd=222
obd_cv = .068 #COEFFICIENT OF VARIATION
#REGRESSION CONSTANTS
b0 = -19.8198
b1 = -0.1169
b2 = 1.0414
b3 = .0782
#CALCULATE BULK DENSITY FROM INORGANIC PERCENTAGE AND ORGANIC BULK DENSITY
bd = obd/(1-ioc/100)

#CONVERT FROM VOLUMETRIC TO GRAVIMETRIC MOISTURE CONTENT (PERCENT)
def v2g(vmc):
    #FACTOR OF 100 IS FOR FRACTION -> PERCENT
    #FACTOR OF 1000 IS FOR CONVERSION FROM 1 g/cm^3 TO 1 kg/m^3
    gmc = 1000*(100*vmc/bd)
    return gmc

#IGNITION PROBABILITY (LOGISTIC MODEL)
def p(gmc, ioc, obd):
    prob = 1/(1+np.exp(-(b0 + b1*gmc + b2*ioc + b3*obd))) 
    return prob

#PLOT PROBABILITY CURVE
x = np.linspace(0,.48,30) #SET UP SOIL MOISTURE VALUES (cm^3/cm^3)
y = p(v2g(x), ioc, obd)
yhigh = p(v2g(x), ioc*(1+ioc_cv), obd*(1+obd_cv))
ylow = p(v2g(x), ioc*(1-ioc_cv), obd*(1-obd_cv))

#PLOT PARAMETERS
ax2=axarr[1].twinx()
ax2.plot(x,y,'r',linestyle='--', linewidth=3)

#PLOT UNCERTAINTY BOUNDS
ax2.fill_between(x,ylow,yhigh,facecolor='pink',interpolate=True, alpha=0.35)
ax2.set_xlabel('Soil moisture (cm$^3$/cm$^3$)')
ax2.set_ylabel('Ignition Probability',color='red')
ax2.tick_params('y',colors='r', length=7, width=2)
ax2.set_ylim((0,1))
ax2.set_xlim((0.1,.5))

#PLOT PARAMETERS
sns.despine(ax=axarr[0], offset=10)
sns.despine(ax=axarr[1], offset=10)
sns.despine(ax=ax2, offset=10, right=False,left=False)
ax2.yaxis.tick_right()

ftsz=26

for item in ([axarr[0].title, axarr[0].xaxis.label, axarr[0].yaxis.label] + axarr[0].get_xticklabels() + axarr[0].get_yticklabels()):
    item.set_fontsize(ftsz)
    
axarr[1].tick_params(axis='both', which='major', labelsize=ftsz, length=7, direction='in')
ax2.tick_params(axis='both', which='major', labelsize=ftsz, length=7, direction='in')

for item in ([axarr[1].title, axarr[1].xaxis.label, axarr[1].yaxis.label] + axarr[1].get_xticklabels() + axarr[1].get_yticklabels()):
    item.set_fontsize(ftsz)

for item in ([ax2.title, ax2.xaxis.label, ax2.yaxis.label] + ax2.get_xticklabels() + ax2.get_yticklabels()):
    item.set_fontsize(ftsz)

#PLOT A AND B LABELS
fig.text(0,.93,'a', fontsize=ftsz, weight='bold')
fig.text(.46,.93,'b', fontsize=ftsz, weight='bold')
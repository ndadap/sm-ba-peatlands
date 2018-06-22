
"""
Purpose: Plot soil moisture-burned area relationship at longer lead times.

Contact: Nathan Dadap, ndadap@stanford.edu
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#variables: fire and sm (with lead time)
fire = np.load('G:/My Drive/data/gfed_fire/gfed_numpy_tables/fire_small_2015-2016_daily.npy')
firedates = pd.date_range('1/1/2015','12/31/2016', freq='D')
sm = np.load('G:/My Drive/data/smap_vod_sm/smap_numpy_tables/MTDCA_0.25deg_daily.npy')
smdates = pd.date_range('4/1/2015','3/31/2017', freq='D')

leadtimes=[50,70,90]

for idx,lead in enumerate(leadtimes):
    #get a copy of sm matrix, subset end of matrix and fill front of timeseries (based on lead time) with nans
    lag_var = np.concatenate((np.full((lead,56,103),np.nan),sm[:(731-lead),:,:]),axis=0)
    lag_var[:lead] = np.nan
    if idx == 0:
        sm10 = lag_var
    if idx == 1:
        sm20 = lag_var
    if idx == 2:
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

#remove unrealistic 0 values from soil moisture dataset
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
#sum the fire, and average the soil moisture
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


#subset to peatlands only
peatmask = np.load('G:/My Drive/data/peat_map/numpy_tables/peatarea_0.25deg.npy') #peat=1, nonpeat=nan
peatmask = peatmask[331:387,1097:1200]
##invert mask for non-peatlands only
#peatmask[np.isnan(peatmask)] = 0.5 
#peatmask[peatmask==1] = np.nan
#peatmask[peatmask==0.5] = 1
def apply_peatmask(datain):
    for day in range(len(datain[:,0,0])):
        datain[day,:,:] = datain[day,:,:] * peatmask
    return datain

sum_fire = apply_peatmask(sum_fire)
avg_sm = apply_peatmask(avg_sm)
avg_sm10 = apply_peatmask(avg_sm10)
avg_sm20 = apply_peatmask(avg_sm20)
avg_sm30 = apply_peatmask(avg_sm30)


#flatten
f = np.ndarray.flatten(sum_fire)
sm = np.ndarray.flatten(avg_sm)
sm10 = np.ndarray.flatten(avg_sm10)
sm20 = np.ndarray.flatten(avg_sm20)
sm30 = np.ndarray.flatten(avg_sm30)


#PLOTTING
#some formatting
ftsz=22
sns.set_style("white")
sns.set_style("white", {'xtick.major.size': 3.0,'ytick.major.size': 3.0}) #tick size
sns.set_style({"xtick.direction": "in","ytick.direction": "in"}) #tick direction

plt.rc('xtick',labelsize=ftsz-2)
plt.rc('ytick',labelsize=ftsz-2)

#establish bins
realval_sm = np.sort(sm[~np.isnan(sm)])
num_bins = 8
intervals = np.full((num_bins),np.nan)
for i in range(num_bins):
    intervals[i] = realval_sm[int(len(realval_sm)*i/num_bins)]
intervals = np.append(intervals, 0.57661124700231015)
intervals = np.append(0, intervals)

#iterate over each lead time
for idx,var in enumerate([sm,sm10,sm20,sm30]):
    
    #using histogram, we get the weight of each bin 
    weighted = np.histogram(var,bins=intervals,weights=f,density=False, range=[min(intervals),max(intervals)])[0]
    non_weighted = np.histogram(var,bins=intervals,density=False,range=[min(intervals),max(intervals)])[0]
    height = weighted/non_weighted #this gives us the mean
    bins = intervals            
    bin_centers = (bins[1:]+bins[:-1])*0.5
    
    #get standard deviation and count in each bin
    stdvs = np.full((len(bin_centers)),np.nan)
    count = np.full((len(bin_centers)),np.nan)
    sterr = np.full((len(bin_centers)),np.nan)
    height2 = np.full((len(bin_centers)),np.nan) #for comparison
    for num,x in enumerate(bin_centers):
        #create mask to get only values from one bin at a time
        bin_mask = [(var > bins[num]) & (var < bins[num+1])]
        var_in_bin = var[bin_mask]
        fire_in_bin = f[bin_mask]
        
        #get std and count of values in that bin        
        stdvs[num] = np.nanstd(fire_in_bin)
        def rlct(vector): return np.count_nonzero(~np.isnan(vector)) #count num of real values
        count[num] = rlct(var_in_bin*fire_in_bin)
        sterr[num] = np.nanstd(fire_in_bin)/(rlct(var_in_bin*fire_in_bin))**0.5
        #height2[num] = np.nansum(fire_in_bin)/count[num] #for comparison to weight method

    #plotting
    line = ['-',"--","-.",":"]
    lab = ['no lead',str(leadtimes[0])+'-day lead', str(leadtimes[1])+'-day lead', str(leadtimes[2])+'-day lead']
    
    if idx==0:
        fig, ax = plt.subplots(figsize=(5,5))
#        fig.dpi=300
        ax.plot(bin_centers,height, label=lab[idx], linewidth=3, ls=line[idx])
        continue
    ax.plot(bin_centers,height, label=lab[idx], linewidth=2, ls=line[idx])
    #ax.fill_between(bin_centers, height+sterr, height-sterr, alpha=0.5)

ax.set_xlabel("Soil moisture (cm$^3$/cm$^3$)")
ax.set_ylabel("Fractional burned area")

ax.legend(fontsize=ftsz)
ax.set_xticks(ticks=np.arange(0.1,0.51,0.1)) #tick locations and interval
ax.set_yticks(ticks=np.arange(0,0.0151,0.0025))

ax.set_ylim((0,.015))
ax.set_xlim((0.1,.5))
    
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(ftsz)

sns.despine(offset=10)







#============================================DATA SETUP========================================
"""
Purpose: Setup dataframe "df" with the following columns:
- latitude index
- longitude index
- day of year
- average soil moisture with 10-day lead
- precipitation sum over prior 90 days
- fire (burned area) summed over 10-day window

Contact: Nathan Dadap, ndadap@stanford.edu
"""

import numpy as np
import pandas as pd

#ISEA AREA PIXEL RANGE (INDICES)
latsize = 56
lonsize = 103


########## PREPARE PRECIP DATA ##########
precip= np.load('G:/My Drive/data/trmm_daily/numpy_tables/200301-201612_TRMM_0.25deg_90day_priorsum.npy')

#SUBSET FOR 2015-2016 (150 DAYS PER DEFINITION OF DRY SEASON)
p2yr = np.empty((2*150,56,103))

#SUBSET DRY SEASON ONLY
df=pd.DataFrame({'dates':pd.date_range('1/1/2003','12/31/2016', freq='D')})
df['doy'] = df['dates'].dt.dayofyear
df['year'] = df['dates'].dt.year
df2 = df.loc[(df.doy>=152) & (df.doy<302) & (df.year>2014)]
1
for idx, val in enumerate(df2.index):
    p2yr[idx,:,:] = precip[val,:,:]

#SELECT PRECIPITATION VALUES FROM FIRST DAY OF EACH 10-DAY WINDOW
def firstval(var):
    firstdim = int(len(var[:,0,0])/10)
    newmat = np.full((firstdim,latsize,lonsize), np.nan)
    for i in range(firstdim):
        newmat[i,:,:] = var[i*10,:,:]
    return newmat
p = firstval(p2yr)


########## PREPARE SOIL MOISTURE AND FIRE DATA ##########

#LOAD VARIABLES
fire = np.load('G:/My Drive/data/gfed_fire/gfed_numpy_tables/fire_small_2015-2016_daily.npy')
sm = np.load('G:/My Drive/data/smap_vod_sm/smap_numpy_tables/MTDCA_0.25deg_daily.npy')

#CREATE DATE VECTORS ASSOCIATED WITH DATERANGES OF DATA
firedates = pd.date_range('1/1/2015','12/31/2016', freq='D')
smdates = pd.date_range('4/1/2015','3/31/2017', freq='D')

#GET 10-DAY LEAD SOIL MOISTURE DATA
lead=10
lag_var = np.concatenate((np.full((lead,latsize,lonsize),np.nan),sm[:(731-lead),:,:]),axis=0)
lag_var[:lead] = np.nan
sm10 = lag_var

#KEEP DATA FROM UNION OF DATASET DATERANGES
start_date, end_date = pd.to_datetime('2014-01-01'), pd.to_datetime('2018-01-01') #INITIALIZE DATES WITH RANGE THAT ENCOMPASSES BOTH VARIABLES
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
fire_trimmed = date_trim(fire, firedates)
sm10_trimmed = date_trim(sm10, smdates)

#SET ZERO VALUES FROM SOIL MOISTURE DATA TO NAN (ERRONEOUS RETRIEVAL)
def no0(dataset):
    dataset[dataset==0] = np.nan
    return dataset
sm10_trimmed = no0(sm10_trimmed)

#TAKE SUBSET FROM DRY SEASON ONLY
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
sm10_trimmed = dry_subset(sm10_trimmed)

#SCALE TO 10-DAY WINDOWS (AVERAGE SOIL MOISTURE, SUM OF FIRE)
def avg10day(var):
    firstdim = int(len(var[:,0,0])/10)
    avg = np.full((firstdim,latsize,lonsize), np.nan)
    for n in np.arange(1,firstdim+1):
        startidx = n*10-10
        endidx = n*10
        avg[n-1,:,:] = np.nanmean(var[startidx:endidx,:,:],axis=0)
    return avg
def sum10day(var):
    firstdim = int(len(var[:,0,0])/10)
    total = np.full((firstdim,latsize,lonsize), np.nan)
    for n in np.arange(1,firstdim+1):
        startidx = n*10-10
        endidx = n*10
        total[n-1,:,:] = np.nansum(var[startidx:endidx,:,:],axis=0)
    return total
sum_fire = sum10day(fire_trimmed)
avg_sm10 = avg10day(sm10_trimmed)


########## SCALING FOR PREDICTOR VARIABLES ##########
#NEEDS SCALING FOR LINEAR REGRESSION CURVE FITTING FUNCTION
#BRUTE FORCE METHOD DERIVED OPTIMAL SCALAR FOR HIGHEST MEAN R2
pscalar = 1
scaled_p = p/pscalar 
smscalar = 1
scaled_sm = avg_sm10/ smscalar


########## SETUP METADATA ARRAYS (LAT, LON, PEAT) ##########

#PEAT ARRAY (HIGH TRANSMISSIVITY ONLY)
peat_area = np.load('G:/My Drive/data/peat_map/numpy_tables/high_transmissivity_peatarea_0.25deg.npy')
#PEAT ARRAY (ANY TRANSMISSIVITY)
#peat_area = np.load('G:/My Drive/data/peat_map/numpy_tables/peatarea_0.25deg.npy')
#peat_area = peat_area[331:387,1097:1200]

#BURNED AREA QUARTILE ARRAY
mean = np.load('G:/My Drive/scripts/paper_plots/sm_ba/aic/mean_ba.npy')

#LOAD PEAT MASK
#peat_area = np.load('G:/My Drive/data/peat_map/numpy_tables/high_transmissivity_peatarea_0.25deg.npy')

#CALCULATE MEAN BURNED AREA OVER ALL TIME
mean_ba = np.nansum(mean,axis=0)
mean_ba[np.isnan(np.nanmean(avg_sm10,0))] == np.nan
mean_ba = mean_ba*peat_area #SUBSET TO PEAT AREAS ONLY

#GET BURNED AREA QUARTILE SIZES
flat_ba = mean_ba[~np.isnan(mean_ba)].flatten()
l1 = np.percentile(flat_ba,25)
l2 = np.percentile(flat_ba,50)
l3 = np.percentile(flat_ba,75)

#CREATE BURNED AREA QUARTILE MASKS // 4th quartile is largest mean burned area, 1st is smallest
qmap = np.full((56,103), np.nan)
for lat in range(56):
    for lon in range(103):
        if mean_ba[lat,lon]<l1:
            qmap[lat,lon] = 1
        elif (mean_ba[lat,lon]>=l1) & (mean_ba[lat,lon]<l2):
            qmap[lat,lon] = 2
        elif (mean_ba[lat,lon]>=l2) & (mean_ba[lat,lon]<l3):
            qmap[lat,lon] = 3
        elif mean_ba[lat,lon]>l3:
            qmap[lat,lon] = 4

#CREATE FLAT ARRAYS
latidx = np.arange(latsize).reshape(latsize,1).repeat(lonsize,axis=1)
lonidx = np.arange(lonsize).reshape(1,lonsize).repeat(latsize,axis=0)
#EXTEND IN TIME DIMENSION (AXIS=0)
latidx = latidx.reshape(1,latsize,lonsize).repeat(len(sum_fire[:,0,0]),axis=0)
lonidx = lonidx.reshape(1,latsize,lonsize).repeat(len(sum_fire[:,0,0]),axis=0)
peat = peat_area.reshape(1,latsize,lonsize).repeat(len(sum_fire[:,0,0]),axis=0)

########## SETUP DATAFRAME ##########

#FLATTEN FOR ENTRY INTO COLUMN
flat_sum_fire = sum_fire.flatten()
flat_avg_sm10 = scaled_sm.flatten()
flat_p = scaled_p.flatten()
flat_lat = latidx.flatten()
flat_lon = lonidx.flatten()
flat_peat = peat.flatten()

#CREATE DATES VECTOR
dates15 = pd.date_range(start='2015-06-01', end='2015-10-19',freq='10D')
dates = dates15.append(pd.date_range(start='2016-05-31', end='2016-10-18',freq='10D')) #subtracted one day because of leap year
dates = dates.repeat(latsize*lonsize) #repeated using area of grid

#CREATE DATAFRAME
df = pd.DataFrame({'dates':dates, 'latidx':flat_lat, 'lonidx':flat_lon, 'peat':flat_peat, 'fire':flat_sum_fire, 'sm10':flat_avg_sm10, 'precip':flat_p})

#REMOVE POINTS WHEN THERE IS A MISSING VALUE (SET TO NAN)
df.dropna(inplace=True)

#SAVE FOR USE IN REGRESSION SCRIPT
df.to_pickle('data.pkl')













# ====================================== REGRESSION ===================================














"""
Calculating coefficient of variation (R^2) for burned area regression with soil moisture and with precipitation. 

Contact: Nathan Dadap, ndadap@stanford.edu
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import scipy.stats

#ISEA AREA PIXEL RANGE (INDICES)
latsize = 56
lonsize = 103


########## FUNCTIONS ##########

#PIECEWISE FORM
def piecewise_trainer(vector_in, a,b,c):
    out=np.full((len(vector_in)),np.nan)
    for idx, point in enumerate(vector_in):
        if point <= alpha:
            out[idx] = a + b*point
        if point > alpha:
            out[idx] = a + b*point + c*(point-alpha)
    return out

#MULTIPLE LINEAR
def combine_trainer(preds, a,b,c,d):
    sm_preds = preds[0]
    precip_preds = preds[1]
    return a + b*sm_preds + c*precip_preds + d*sm_preds*precip_preds

#R-SQUARED CALCULATION
def rsqrd(data_vector, prediction_vector):
    t1, t2, r, t4, t5 = scipy.stats.linregress(data_vector, prediction_vector)
    #r=((1/len(data_vector))*np.sum((data_vector-data_vector.mean())*(prediction_vector-prediction_vector.mean())) / (prediction_vector.std()*data_vector.std()))
    return r**2



########## CALCULATE RSQUARED MAP ##########

#########
import matplotlib.pyplot as plt
from sklearn import preprocessing
#LOAD DATA
df = pd.read_pickle('data.pkl')
#data columns: 'dates', 'latidx', 'lonidx', 'fire', 'sm10', 'precip'

#INITALIZE MAP FOR BOTH VARIABLES
sm_rsquared_map = np.full((latsize,lonsize),np.nan)
precip_rsquared_map = np.full((latsize,lonsize),np.nan)
multi_rsquared_map = np.full((latsize,lonsize),np.nan)

#SM PARAMETERS
a_map_sm = np.full((latsize,lonsize),np.nan)
b_map_sm = np.full((latsize,lonsize),np.nan)
c_map_sm = np.full((latsize,lonsize),np.nan)
alpha_map_sm = np.full((latsize,lonsize),np.nan)

#PRECIP PARAMETERS
a_map_p = np.full((latsize,lonsize),np.nan)
b_map_p = np.full((latsize,lonsize),np.nan)
c_map_p = np.full((latsize,lonsize),np.nan)
alpha_map_p = np.full((latsize,lonsize),np.nan)

#COMBINATION PARAMETERS
a_map_com = np.full((latsize,lonsize),np.nan)
b_map_com = np.full((latsize,lonsize),np.nan)
c_map_com = np.full((latsize,lonsize),np.nan)
d_map_com = np.full((latsize,lonsize),np.nan)

#test pixels for plotting
test_pts = [(41,77),(37,78),(38,81), (40,75),(33,63),(39,44), (42,45),(45,45),(39,79), (26,72)]


#KEEPING TRACKING OF SCRIPT
import sys
counter=0
missedYs=[]
missedXs=[]

#ITERATE OVER ALL PIXELS IN ISEA
for yidx in range(latsize):
    for xidx in range(lonsize):

        ######################### GET DATA, SKIP PIXEL IF INSUFFICIENT DATA FOR THIS PIXEL
        #ENSURE SUFFICIENT SAMPLES
        test = df.loc[(df.latidx == yidx) & (df.lonidx == xidx)]        
                
        #SKIP PIXEL IF INSUFFICIENT TEST DATA
        if len(test.fire)<10:
            continue
        
        #SELECT TRAINING DATA (5X5 PIXEL WINDOW)
        window_size = 5
        idx_diff = 3
        train = df.loc[(df.latidx > (yidx-idx_diff)) & (df.latidx < (yidx+idx_diff)) & (df.lonidx > (xidx-idx_diff)) & (df.lonidx < (xidx+idx_diff))]
        #########################
        
        
        ######################### RESCALE DATA
        column_names = ['fire', 'precip', 'sm10']
        lats = train.latidx
        lons = train.lonidx
        x = train[column_names].values #returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        new = pd.DataFrame(x_scaled)
        new.columns = column_names
        new['latidx'] = np.array(lats)
        new['lonidx'] = np.array(lons)
        train = new
        test = train.loc[(train.latidx == yidx) & (train.lonidx == xidx)]
        
        #REPACKAGE DATA INTO LIST
        train_ind = [train.sm10, train.precip]
        #########################
        
        
        ######################### SOIL MOISTURE REGRESSION
        #INITIALIZE 
        best_alpha = 0
        best_a = 0
        best_b = 0
        best_c = 0
        best_prediction = np.empty((0))
        highest_r2 = 0
        
        #specify optimization parameter bounds and method
        bounds = ([0, -np.inf, 0], [np.inf, 0, np.inf])
        method = 'trf'
        
        #TRY DIFFERENT BREAKPOINTS AND INITIAL GUESSES
        for b_guess in [0, -1, -10, -50]:
            for c_guess in [0, 1, 10, 50]:
                for alpha in np.linspace(train.sm10.min(), train.sm10.max(), 30):
                        
                    #FIND OPTIMAL PARAMETERS
                    sm_opt, cov = curve_fit(f = piecewise_trainer,
                                            xdata = train.sm10,
                                            ydata = train.fire,
                                            p0 = [0.5, b_guess, c_guess],
                                            method = method,
                                            bounds = bounds)
        
                    #CALCULATE PREDICTION VECTOR
                    sm_prediction = piecewise_trainer(test.sm10, sm_opt[0], sm_opt[1], sm_opt[2])
                    
                    #INVALID OPTIMIZATION IF SLOPE IS POSITIVE
                    if sm_opt[1]+sm_opt[2] > 0:
                        continue
                    
                    #CALCULATE R2
                    current_r2 = rsqrd(sm_prediction, test.fire)
        
                    #CHANGE PARAMETERS IF R2 IS HIGHER
                    if current_r2 > highest_r2:
                        
                        #SET BEST PARAMETERS
                        best_alpha = alpha
                        best_a = sm_opt[0]
                        best_b = sm_opt[1]
                        best_c = sm_opt[2]
                        
                        #SET BEST PREDICTION
                        best_prediction = sm_prediction
                        
                        #UPDATE HIGHEST R2 VALUE
                        highest_r2 = current_r2
                        
        #SET VARS
        sm_prediction= best_prediction
        a_map_sm[yidx,xidx] = best_a
        b_map_sm[yidx,xidx] = best_b
        c_map_sm[yidx,xidx] = best_c
        alpha_map_sm[yidx,xidx] = best_alpha
        
        #### FOR TEST PLOT ####
        alpha = alpha_map_sm[yidx,xidx]
        sms = np.linspace(train.sm10.min(),train.sm10.max(), 30)
        BAsms = piecewise_trainer(sms, best_a, best_b, best_c)
        #########################
        
        
        ######################### PRECIP SINGLE REGRESSION
        best_alpha = 0
        best_a = 0
        best_b = 0
        best_c = 0
        best_prediction = np.empty((0))
        highest_r2 = 0
        
        #TRY DIFFERENT BREAKPOINTS AND INITIAL GUESSES
        for b_guess in [0, -1, -10, -50]:
            for c_guess in [0, 1, 10, 50]:
                for alpha in np.linspace(train.precip.min(), train.precip.max(), 30):
            
                    #FIND OPTIMAL PARAMETERS
                    p_opt, cov = curve_fit(f = piecewise_trainer,
                                           xdata = train.precip,
                                           ydata = train.fire,
                                           p0 = [0.5, b_guess, c_guess],
                                           method = method,
                                           bounds = bounds)
                    
                    #CALCULATE PREDICTION VECTOR
                    precip_prediction = piecewise_trainer(test.precip, p_opt[0], p_opt[1], p_opt[2])
                    
                    #INVALID OPTIMIZATION IF SLOPE IS POSITIVE
                    if p_opt[1] + p_opt[2] > 0:
                        continue
                    
                    #CALCULATE R2
                    current_r2 = rsqrd(precip_prediction, test.fire)
        
                    #CHANGE PARAMETERS IF R2 IS HIGHER
                    if current_r2 > highest_r2:
                        
                        #SET BEST PARAMETERS
                        best_alpha = alpha
                        best_a = p_opt[0]
                        best_b = p_opt[1]
                        best_c = p_opt[2]
                        
                        #SET BEST PREDICTION
                        best_prediction = precip_prediction
                        
                        #UPDATE HIGHEST R2 VALUE
                        highest_r2 = current_r2
        
        #SET VARS
        precip_prediction= best_prediction
        a_map_p[yidx,xidx] = best_a
        b_map_p[yidx,xidx] = best_b
        c_map_p[yidx,xidx] = best_c
        alpha_map_p[yidx,xidx] = best_alpha
        
        #### FOR TEST PLOT ####
        alpha = alpha_map_p[yidx,xidx]
        precips = np.linspace(train.precip.min(),train.precip.max(), 30)
        BAprecips = piecewise_trainer(precips, a_map_p[yidx,xidx], b_map_p[yidx,xidx], c_map_p[yidx,xidx])
        #########################
        
        
        ######################### MULTIPLE REGRESSION WITH INTERACTION TERM
        #PACKAGE PREDICTIONS INTO ONE VARIABLE
        preds = [sm_prediction, precip_prediction]
        
        #GET OPTIMAL COMBINATION PARAMETERS
        best_prediction = np.empty((0))
        highest_r2 = 0
        best_a = 0
        best_b = 0
        best_c = 0
        best_d = 0
        
        for b_guess in [-1, 0, 1]:
            for c_guess in [-1, 0, 1]:
                for d_guess in [-1, 0, 1]:

                    #CALCULATE OPTIMUM PARAMETERS
                    try:
                        com_opt, covs = curve_fit(f = combine_trainer,
                                                  xdata = preds,
                                                  ydata = test.fire,
                                                  method = method,
                                                  p0 = [0, b_guess, c_guess, d_guess])
                    except ValueError:
                        continue
        
                    #COMBINED PREDICTION
                    multi_prediction = combine_trainer(preds, com_opt[0], com_opt[1], com_opt[2], com_opt[3])
                    
                    #CALCULATE R2
                    current_r2 = rsqrd(multi_prediction, test.fire)
                    
                    #UPDATE PARAMETERS IF R2 IS HIGHER
                    if current_r2 > highest_r2:
                        
                        #SET BEST PARAMETERS
                        best_a = com_opt[0]
                        best_b = com_opt[1]
                        best_c = com_opt[2]
                        best_d = com_opt[3]
                        
                        #SET BEST PREDICTION
                        best_prediction = multi_prediction
                                                
                        #UPDATE HIGHEST R2 VALUE
                        highest_r2 = current_r2
        
                    
    
        #SAVE BEST PARAMETERS       
        a_map_com[yidx,xidx] = best_a
        b_map_com[yidx,xidx] = best_b
        c_map_com[yidx,xidx] = best_c
        d_map_com[yidx,xidx] = best_d
        
        #### FOR TEST PLOT ####
        BAcombined_sm = combine_trainer([BAsms,np.repeat(np.nanmedian(BAprecips),len(BAsms))],
                                        best_a, best_b, best_c, best_d)
        BAcombined_p = combine_trainer([np.repeat(np.nanmedian(BAsms),len(BAsms)), BAprecips],
                                       best_a, best_b, best_c, best_d)
        #########################
        
        
        ######################### CALCULATE R2S
        #SKIP ITERATION IF NO PREDICTION IS MADE
        if len(sm_prediction) == 0 or len(precip_prediction) == 0:
            print('missed point: ('+str(yidx)+', '+str(xidx)+')')
            missedYs.append(yidx)
            missedXs.append(xidx)
            continue
        
        sm_rsquared_map[yidx,xidx] = rsqrd(test.fire, sm_prediction) #SOIL MOISTURE
        precip_rsquared_map[yidx,xidx] = rsqrd(test.fire, precip_prediction) #PRECIPITATION
        multi_rsquared_map[yidx,xidx] = rsqrd(test.fire, multi_prediction) #MULTIPLE VARS
        #########################
        
        ######################### TEST PLOTS/INFO BELOW      
        # sm and precip multiple regression to plot
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))
        ax1.scatter(train.sm10, train.fire, color='b', label='training') #train data
        ax1.scatter(test.sm10, test.fire, color='orange', label='pixel only') #test data
        ax1.plot(sms, BAsms, color = 'red', label='single', linestyle=':')
        #ax1.plot(test.sm10, sm_prediction, color = 'red', label='single sm', linestyle='-.')
        ax1.plot(sms, BAcombined_sm, color='red', label='multi')
        ax1.axvline(x= alpha_map_sm[yidx,xidx] * smscalar, color = 'black', linestyle='--')
        ax1.legend()
        ax1.set_title('sm')
        
        ax2.scatter(train.precip, train.fire, color='b', label='training') #train data
        ax2.scatter(test.precip, test.fire, color='orange', label='pixel only') #test data
        ax2.plot(precips, BAprecips, color = 'red', label='single', linestyle=':')
        ax2.plot(precips, BAcombined_p, color='red', label='multi')
        ax2.axvline(x= alpha_map_p[yidx,xidx], color = 'black', linestyle='--')
        ax2.legend()
        ax2.set_title('precip')
        
        #
        fig.suptitle('latidx: '+str(yidx)+', lonidx: '+str(xidx))
        plt.show()


np.save('G:/My Drive/scripts/paper_plots/sm_ba/aic/sm_rsquared_map.npy', sm_rsquared_map)
np.save('G:/My Drive/scripts/paper_plots/sm_ba/aic/precip_rsquared_map.npy', precip_rsquared_map)
#======================================== PLOTTING ==================================














"""
FIGURE 4

Left panel: Soil moisture and precipitation R^2 comparison by time-averaged burned area quartile
Right panel: Delta R^2 map

Contact: Nathan Dadap, ndadap@stanford.edu
"""


from mpl_toolkits.basemap import Basemap
import numpy as np
import pandas as pd
from matplotlib import gridspec
import seaborn as sns


#LOAD RSQUARED MAPS
sm_map = sm_rsquared_map
precip_map = precip_rsquared_map
multi_map = multi_rsquared_map

#LOAD BURNED AREA
mean = np.load('G:/My Drive/scripts/paper_plots/sm_ba/aic/mean_ba.npy')

#LOAD PEAT MASK
#peat_area = np.load('G:/My Drive/data/peat_map/numpy_tables/high_transmissivity_peatarea_0.25deg.npy')

#CALCULATE MEAN BURNED AREA OVER ALL TIME
mean_ba = np.nansum(mean,axis=0)
mean_ba[np.isnan(sm_map)] == np.nan
mean_ba = mean_ba*peat_area #SUBSET TO PEAT AREAS ONLY

#GET BURNED AREA QUARTILE SIZES
flat_ba = mean_ba[~np.isnan(mean_ba)].flatten()
l1 = np.percentile(flat_ba,25)
l2 = np.percentile(flat_ba,50)
l3 = np.percentile(flat_ba,75)

#CREATE BURNED AREA QUARTILE MASKS // 4th quartile is largest mean burned area, 1st is smallest
qmap = np.full((56,103), np.nan)
for lat in range(56):
    for lon in range(103):
        if mean_ba[lat,lon]<l1:
            qmap[lat,lon] = 1
        elif (mean_ba[lat,lon]>=l1) & (mean_ba[lat,lon]<l2):
            qmap[lat,lon] = 2
        elif (mean_ba[lat,lon]>=l2) & (mean_ba[lat,lon]<l3):
            qmap[lat,lon] = 3
        elif mean_ba[lat,lon]>l3:
            qmap[lat,lon] = 4


#SETUP DATAFRAME TO STORE R^2, REGRESSOR VARIABLE, AND QUARTILE
df2plot = pd.DataFrame(columns=['R^2','Regressor Variable','Quartile'])
i=0
for idx, variable in enumerate([sm_map, precip_map, multi_map]):
    #ASSIGN PREDICTOR VARIABLE
    if idx==0: var='Soil Moisture'
    elif idx==1: var='Precipitation'
    elif idx==2: var='Multi'
    
    for lat in range(56):
        for lon in range(103):
            #SKIP IF NOT IN PEATLANDS
            if np.isnan(mean_ba[lat,lon]):
                continue
            
            #BUILD DATAFRAME BY ROW
            df2plot.loc[i] = [variable[lat,lon], var, qmap[lat,lon]]
            
            #ADVANCE TO NEXT ROW
            i+=1

#CALCULATE DELTA R-SQUARED
delta_rsq = sm_map - precip_map



########### PLOTTING

#PLOT VISUALIZATION PARAMETERS
fig=plt.figure(figsize=(14,5))
sns.set_style("white")
sns.set_style("ticks")
gs = gridspec.GridSpec(1, 2, width_ratios=[5, 8])
gs.update(wspace=0.03)

ax1=plt.subplot(gs[1])
ax2=plt.subplot(gs[0])


#SETUP FIGURE GRID
import os
os.chdir('G:/My Drive/data/gfed_fire/gfed_numpy_tables/')
fire_lat=np.load('fire_lat.npy')
fire_lon=np.load('fire_lon.npy')
x,y=np.meshgrid(fire_lon,fire_lat)

#LOAD DATA
plotter = np.zeros((720,1440))
plotter[331:387,1097:1200] = peat_area * delta_rsq #SUBSET TO PEATLANDS ONLY

#SETUP BASEMAP
m = Basemap(projection='cyl', llcrnrlat=-6, urcrnrlat=7,llcrnrlon=94.5, urcrnrlon=119, resolution='h', lon_0=0, ax=ax1)

# PLOT DATA ONTO MAP
import matplotlib as mpl
cmap1=mpl.colors.LinearSegmentedColormap.from_list("", ["blue","lightgrey","red"]) #bwr
cmap1.set_under('white', 1.)

#IMPORT DATA, USE VISUALIZATION PARAMETERS
offset = 0.125 #PIXEL SHIFT
rng = 0.4 #RANGE
ftsz = 24 #FONT SIZE
image = m.pcolormesh(x-offset,y+offset,plotter,shading='flat', cmap=cmap1,vmin=-rng, vmax=rng)

#DRAW COASTLINES AND MAP BOUNDARY
m.drawcoastlines()
m.drawmapboundary()

#PLOT COLORBAR
cbaxes = fig.add_axes([.47, .145, 0.51, 0.02]) 
cb=plt.colorbar(image, cax =cbaxes, ax=ax1, orientation="horizontal", ticks=[-rng,-rng/2,0,rng/2,rng])             
cb.ax.tick_params(labelsize=ftsz-2)
cb.ax.set_xlabel('$\Delta$R$^2$ in peatlands',fontsize=ftsz)

#BOXPLOT
my_palette = {'Soil Moisture':'red', "Precipitation":'blue', "Multi":'orange'}
sns.boxplot(x="Quartile", y="R^2", hue="Regressor Variable", data=df2plot, palette=my_palette, ax=ax2, fliersize=0)

#BOXPLOT VISUALIZATION PARAMETERS
ax2.set_xlabel('Pixel-average burned area quartile', fontsize=ftsz)
ax2.set_ylabel('R$^2$ (fire, regressor)', fontsize=ftsz)
ax2.legend(handles=ax2.get_legend_handles_labels()[0],labels=['SM','Precip','Multi'], fontsize=ftsz-2)
ax2.set_xticklabels(labels=['q1','q2','q3','q4'])
ax2.tick_params(axis='both', labelsize=ftsz-2)
for patch in ax2.artists:
    r, g, b, a = patch.get_facecolor() 
    patch.set_facecolor((r, g, b, .6)) 

#LAYOUT
gs.tight_layout(fig)

#A AND B LABEL
fig.text(0.02,.9,'a', fontsize=ftsz, weight='bold')
fig.text(0.43,.9,'b', fontsize=ftsz, weight='bold')

plt.show()



"""

# ====================== INFO ================================
#CALCULATE NUMBER OF PIXELS FOR WHICH SOIL MOISTURE DOES BETTER AND MEAN
len(delta_rsq[delta_rsq>0]) #49
np.mean(delta_rsq[delta_rsq>0]) #.096
#CALCULATE NUMBER OF PIXELS FOR WHICH PRECIP DOES BETTER AND MEAN
len(delta_rsq[delta_rsq<0]) #54
np.mean(delta_rsq[delta_rsq<0]) #-0.107
#MEAN SM R2 IN Q4
df2plot.loc[(df2plot['Quartile']==4) & (df2plot['Regressor Variable']=='Soil Moisture'),:].mean() #.428
#MEAN PRECIP R2 IN Q4
df2plot.loc[(df2plot['Quartile']==4) & (df2plot['Regressor Variable']=='Precipitation'),:].mean() #.427
"""
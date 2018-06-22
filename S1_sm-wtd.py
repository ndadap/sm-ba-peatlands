
"""
SUPPLEMENTARY FIGURE 1: SOIL MOISTURE - WATER TABLE DEPTH RELATIONSHIP

Contact: Nathan Dadap, ndadap@stanford.edu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns

sm_path = 'G:/My Drive/peat_project/Lit Review/sm-wtd_data/mendaram_vwc_data/cleanedup_data/{loc}_datalogger.csv'
ll_path = 'G:/My Drive/peat_project/Lit Review/sm-wtd_data/mendaram_vwc_data/cleanedup_data/{loc}_levelogger.csv'

location='middle'

#load data for site
sm_df = pd.read_csv(sm_path.format(loc=location))
ll_df = pd.read_csv(ll_path.format(loc=location))

#make sure that all vwc and wtd numbers are float type
#turn NANs into NANS that pandas recognizes
sm_df['averageVWC'] = pd.to_numeric(sm_df['averageVWC'], errors='coerce')
ll_df['LEVEL'] = pd.to_numeric(ll_df['LEVEL'], errors='coerce')
#baro_df['level_mh2o'] = pd.to_numeric(baro_df['level_mh2o'], errors='coerce')

#average over each location
sm_avg = sm_df.groupby('TIMESTAMP', as_index=False)['averageVWC'].mean() #NOTE: as_index turns it into a new dataframe
ll_avg = ll_df.groupby('Date', as_index=False)['LEVEL'].mean()
#baro_avg = baro_df.groupby('Date', as_index=False)['level_mh2o'].mean()

#keep only sm values that are overlapping with the levelogger data
sm_vec = sm_avg[sm_avg['TIMESTAMP'].isin(np.array(ll_avg['Date']))]
wt_vec = ll_avg[ll_avg['Date'].isin(np.array(sm_vec['TIMESTAMP']))]
                
#ensure they are in the same order
sm_vec = sm_vec.sort_values(['TIMESTAMP'])
wt_vec = wt_vec.sort_values(['Date'])

#join dataframes
df = pd.merge(left=sm_vec, right=wt_vec, how='inner', left_on='TIMESTAMP', right_on='Date')


############ IDENTIFY TOP OF GROUND SURFACE USING MANN-KENDALL TEST

#find when water table is no longer increasing    
from mk_test import mk_test #function in other script
wt_thresh = 11.2
lowest_p = 1
for depth in np.arange(10.8,11.2,0.01):
    test = mk_test(np.array(df.LEVEL.loc[df.LEVEL>depth]))
    trend = test[0]
    p = test[2]
    if (trend == 'no trend') & (p < lowest_p):
        wt_thresh=depth
        lowest_p = p

#offset data by water table height
offset=wt_thresh

#############

#Linear Regression of SM-WTD
import scipy.stats
reg = scipy.stats.linregress(df.LEVEL-offset, df.averageVWC)
x = np.arange(-0.8,0.2,0.1)
y = reg[0] * x + reg[1]

def wt2sm(input_wtd):
    return reg[0] * input_wtd + reg[1]

############
#FONTSIZE
ftsz = 22
#SETUP PLOT
fig,ax = plt.subplots(figsize=(6,6))
#PLOT SCATTERPLOT
ax.scatter(df['LEVEL']-offset, df['averageVWC'], marker='.', color='black')
ax.set_xlabel('Water table depth (m)', fontsize=ftsz)
ax.set_ylabel('Soil moisture (cm$^3$/cm$^3$)', fontsize=ftsz)
ax.set_xticks(ticks=np.arange(-0.8, 0.1, .2))
ax.tick_params(direction='in', labelsize=20, size=5) #"size" will show tickmarks
    
    
    
    
    
    
    
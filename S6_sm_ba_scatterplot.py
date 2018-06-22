

"""
Purpose: PLOT BURNED AREA VS SOIL MOISTURE SCATTERPLOT AND REGRESSION

Contact: Nathan Dadap, ndadap@stanford.edu
"""

#RUN OTHER SCRIPT TO GET FIRE AND SOIL MOISTURE DATA
from fig3_smba import *

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


#CREATE DATES VECTOR
dates2015 = np.arange('2015-06-05', '2015-10-29', step=10, dtype='datetime64[D]')
dates2016 = np.arange('2016-06-05', '2016-10-29', step=10, dtype='datetime64[D]')
dates = np.concatenate((dates2015,dates2016))
dates = dates.repeat(np.size(avg_sm10[0,:,:])) #SINGLE DAY ARRAY SIZE


########## FIND CALCULATE REGRESSION/PREDICTION
#LINEAR PIECEWISE FORM
def piecewise_trainer(vector_in, a,b,c, alpha):
    out=np.full((len(vector_in)),np.nan)
    for idx, point in enumerate(vector_in):
        if point <= alpha:
            out[idx] = a + b*point
        if point > alpha:
            out[idx] = a + b*point + c*(point-alpha)
    return out

#EXPONENTIAL FORM (UNUSED)
def exponential(x,a1,a2):
    return a1*np.exp(-a2*x)

#LOGISTIC FORM (UNUSED)
def logistic(x,b1,b2,b3):
    return b1/(1+np.exp(-(b2 + b3*x))) 

#CREATE DATAFRAME WITH DATA
df = pd.DataFrame({'sm':sm,'f':f})
df.dropna(inplace=True)

#EXPONENTIAL FIT PARAMETER OPTIMIZATION
opt, cov = curve_fit(exponential, df.sm, df.f) 
a1,a2=opt[0],opt[1]

#LOGISTIC FIT PARAMETER OPTIMIZATION
opt, cov = curve_fit(logistic, df.sm, df.f) 
b1,b2,b3=opt[0],opt[1],opt[2]

#R-SQUARED CALCULATION
import scipy.stats
def rsqrd(data_vector, prediction_vector):
    t1, t2, r, t4, t5 = scipy.stats.linregress(data_vector, prediction_vector)
    return r**2

#INITIALIZE VARIABLES FOR OPTIMIZED PREDICTION AND PARAMETERS
best_prediction = np.empty((0))
highest_r2 = 0
best_c1 = 0
best_c2 = 0
best_c3 = 0
best_alpha = 0

i=1 #TRACKER
#BRUTE FORCE METHOD TO TRYING VARIOUS INITIAL PARAMETERS
for c1_guess in [.01,.03,.05,.1,.15]:
    for c2_guess in [-1.3,-1,-0.7,-0.4,-0.1]:
        print(i)
        i+=1
        for c3_guess in [1.3,1,0.7,0.4,0.1]:
            for c4_guess in [0.15, 0.2, 0.25, 0.3, 0.35]:

                #CALCULATE OPTIMUM PARAMETERS
                opt, cov = curve_fit(f = piecewise_trainer,
                                     xdata = df.sm,
                                     ydata = df.f,
                                     method ='trf',
                                     p0 = [c1_guess, c2_guess, c3_guess, c4_guess])
                
                #COMBINED PREDICTION
                prediction = piecewise_trainer(df.sm, opt[0], opt[1], opt[2], opt[3])
                
                #INVALID OPTIMIZATION IF SLOPE IS POSITIVE
                if opt[1]+opt[2] > 0:
                    continue
                
                #CALCULATE R2
                current_r2 = rsqrd(prediction, df.f)

                #CHANGE PARAMETERS IF R2 IS HIGHER
                if current_r2 > highest_r2:
                    
                    #SET BEST PARAMETERS
                    best_alpha = opt[3]
                    best_c1 = opt[0]
                    best_c2 = opt[1]
                    best_c3 = opt[2]
                    
                    #SET BEST PREDICTION
                    best_prediction = prediction
                    
                    #UPDATE HIGHEST R2 VALUE
                    highest_r2 = current_r2
                            


########### PLOT
#GLOBAL FONT SIZE
ftsz=26

#MODELED LINE
xpts = np.linspace(0.07,0.5,50)
exp_pts = trainer(xpts,a1,a2)
log_pts = trainer2(xpts,b1,b2,b3)
pcw_pts = piecewise_trainer(xpts, best_c1, best_c2, best_c3, best_alpha)

#SETUP PLOT
fig,ax = plt.subplots()
#PLOT POINTS
plot = ax.scatter(x=df.sm, y=df.f, marker='.',label=None)
#PLOT REGRESSION
ax.plot(xpts,pcw_pts, color='red', label='Piecewise Linear Regression')
#LABELS
ax.set_xlabel('Soil moisture (cm$^3$/cm$^3$)', fontsize=ftsz)
ax.set_ylabel('Fractional burned area', fontsize=ftsz)
#BOUNDS
ax.set_ylim([0,.2])
ax.set_xlim([0.05,.45])
ax.set_xticks(np.arange(.05,.5,.1))
#LEGEND
ax.legend(fontsize=ftsz-4)
#FONT SIZE
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(ftsz)    
sns.despine(offset=10)
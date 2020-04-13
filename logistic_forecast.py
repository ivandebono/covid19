import numpy as np
import pandas as pd
from datetime import timedelta
from datetime import datetime
from io import StringIO
from urllib import request as url_request
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from scipy.optimize import curve_fit

# This stuff because pandas or matplot lib complained...
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from scipy import stats,integrate
from scipy.optimize import curve_fit



from scipy import stats,integrate



def logistic(x, L, k, x0, y0):
    """
    General Logistic function.

    Args:
        x    float or array-like, it represents the time
        L    float, the curve's maximum value
        k    float, the logistic growth rate or steepness of the curve.
        x0   float, the x-value of the sigmoid's midpoint
        y0   float, curve's shift in the y axis
    """
    y = L / (1 + np.exp(-k*(x-x0))) + y0
    return y

def logistic_derivative(x, L, k, x0):
    """
    General Gaussian-like function (derivative of the logistic).

    Args:
        x    float or array-like, it represents the time
        L    float, the curve's integral (area under the curve)
        k    float, the logistic growth rate or steepness of the curve.
        x0   float, the x-value of the max value
    """
    y = k * L * (np.exp(-k*(x-x0))) / np.power(1 + np.exp(-k*(x-x0)), 2)
    return y

def fit_curve(curve, ydata, title, ylabel, last_date, coeff_std, do_imgs=False,plt_forecast=False,show_every=5):
    
    xdata = -np.flip(np.arange(len(ydata)))
        
    days_past = -2 # days beyond the start of the data to plot
    days_future = 40 # days after the end of the data to predict and plot
    #show_every = 3 # int value that defines how often to show a date in the x axis. (used not to clutter the axis)    

    myFmt = mdates.DateFormatter('%d/%m') # date formatter for matplotlib
    
    
    total_xaxis = np.array(range(-len(ydata) + days_past, days_future)) + 1


    date_xdata = [last_date + timedelta(days=int(i)) for i in xdata]
    date_total_xaxis = [last_date + timedelta(days=int(i)) for i in total_xaxis]

    future_axis = total_xaxis[len(ydata) - days_past:]
    date_future_axis = [last_date + timedelta(days=int(i)) for i in future_axis]
    
            # Plotting
    fig, ax = plt.subplots(figsize=(15,8))
    ax.xaxis.set_major_formatter(myFmt)
    fig.autofmt_xdate()
    

    start = (len(ydata) - days_past - 1) % show_every
    ax.set_xticks(date_total_xaxis[start::show_every])
    ax.set_xlabel('Date')
    ax.set_ylabel(ylabel)
    ax.set_title(title + ': ' + str(last_date.strftime("%d-%m-%Y")))

    ax.grid(True)

    if curve.__name__ == 'logistic':
        p0_1=ydata[-1]
        p0=[p0_1, 0.5, 1, 0]
        bounds=([0, 0, -100, 0], [2*p0_1, 10, 100, 1])
        #bounds=([0,0,-np.inf,0],[np.inf,np.inf,np.inf,np.inf])
        params_names = ['L', 'k', 'x0', 'y0']
    elif curve.__name__ == 'logistic_derivative':
        p0_1=3*max(ydata) 
        p0=[p0_1, 0.5, 1]
        bounds=([0, 0, -100], [10*p0_1, 10, 100])
        #bounds=([0,0,-np.inf],[np.inf,np.inf,np.inf])
        params_names = ['L', 'k', 'x0']
    else:
        print('this curve is unknown')
        return -1

    popt, pcov = curve_fit(curve, xdata, ydata, p0=p0, bounds=bounds,maxfev=20000)
    
    print(title)
    descr = '    fit: '
    for i, param in enumerate(params_names):
        descr = descr + "{}={:.3f}".format(param, popt[i])
        if i < len(params_names) - 1:
            descr = descr + ', '
    print(descr)
    
    perr = np.sqrt(np.diag(pcov))
    print('perr',perr)
    

    
    pworst = popt + coeff_std*perr
    pbest = popt - coeff_std*perr
    
    
    # Plotting
    # fig, ax = plt.subplots(figsize=(15,8))
#     ax.xaxis.set_major_formatter(myFmt)
#     fig.autofmt_xdate()

    total_xaxis = np.array(range(-len(ydata) + days_past, days_future)) + 1
    date_total_xaxis = [last_date + timedelta(days=int(i)) for i in total_xaxis]
    

    date_xdata = [last_date + timedelta(days=int(i)) for i in xdata]
    
    
    future_axis = total_xaxis[len(ydata) - days_past:]
    date_future_axis = [last_date + timedelta(days=int(i)) for i in future_axis]

    #print('pbest',pbest)
    #print('pworst',pworst)


    
    if plt_forecast==True:
        ax.plot(date_total_xaxis, curve(total_xaxis, *popt), 'g-', label='prediction')
        ax.fill_between(date_future_axis, curve(future_axis, *pbest), curve(future_axis, *pworst), 
        facecolor='red', alpha=0.2, label='std')
    #print('Integral=',np.trapz(curve(total_xaxis, *popt)))
    
    ax.scatter(date_xdata, ydata, color='blue', label='real data',s=8)
    ax.plot(date_xdata, ydata, color='blue',alpha=0.5)



   #  start = (len(ydata) - days_past - 1) % show_every
#     ax.set_xticks(date_total_xaxis[start::show_every])
#     ax.set_xlabel('Date')
#     ax.set_ylabel(ylabel)
#     ax.set_title(title + ': ' + str(last_date.strftime("%d-%m-%Y")))
    ax.legend(loc='upper left')
#     ax.grid(True)

    
    fig=plt.gcf()
    plt.show()
    if do_imgs:
        fig.savefig('plots/'+ title + '.png', dpi=200)
    

    

    return popt, perr

def fit_curve_evolution(curve, ydata1, title, ylabel, last_date, coeff_std, do_imgs=False,plt_forecast=False,
                daily=True,days_past = -2,days_future = 40,show_every = 3,day_start=30):
    
    xdata = -np.flip(np.arange(len(ydata1)))
        
    #days_past = -2 # days beyond the start of the data to plot
    #days_future = 40 # days after the end of the data to predict and plot
    #show_every = 3 # int value that defines how often to show a date in the x axis. (used not to clutter the axis)    

    myFmt = mdates.DateFormatter('%d/%m') # date formatter for matplotlib
    
    
    total_xaxis = np.array(range(-len(ydata1) + days_past, days_future)) + 1


    date_xdata = [last_date + timedelta(days=int(i)) for i in xdata]
    date_total_xaxis = [last_date + timedelta(days=int(i)) for i in total_xaxis]

    future_axis = total_xaxis[len(ydata1) - days_past:]
    date_future_axis = [last_date + timedelta(days=int(i)) for i in future_axis]
    
            # Plotting
    fig, ax = plt.subplots(figsize=(15,8))
    ax.xaxis.set_major_formatter(myFmt)
    fig.autofmt_xdate()
    

    start = (len(ydata1) - days_past - 1) % show_every
    ax.set_xticks(date_total_xaxis[start::show_every])
    ax.set_xlabel('Date')
    ax.set_ylabel(ylabel)
    ax.set_title(title + ': ' + str(last_date.strftime("%d-%m-%Y")))

    ax.grid(True)


    

    coln=len(np.arange(day_start,len(ydata1)+1))
    colors = plt.cm.YlGn(np.linspace(0,1,coln))
    

    
    for j in np.arange(day_start,len(ydata1)+1):
        #print('STEP',j)
        ydata=ydata1[:j]
        xdata = -np.flip(np.arange(len(ydata)))
        #print(len(xdata),len(ydata))

        if curve.__name__ == 'logistic':
            p0_1=ydata[-1]
            p0=[p0_1, 0.5, 1, 0]
            bounds=([0, 0, -100, 0], [2*p0_1, 10, 100, 1])
            #bounds=([0,0,-np.inf,0],[np.inf,np.inf,np.inf,np.inf])
            params_names = ['L', 'k', 'x0', 'y0']
        elif curve.__name__ == 'logistic_derivative':
            p0_1=3*max(ydata) 
            p0=[p0_1, 0.5, 1]
            bounds=([0, 0, -100], [10*p0_1, 10, 100])
            #bounds=([0,0,-np.inf],[np.inf,np.inf,np.inf])
            params_names = ['L', 'k', 'x0']
        else:
            print('this curve is unknown')
            return -1

        popt, pcov = curve_fit(curve, xdata, ydata, p0=p0, bounds=bounds,maxfev=20000)
     
        perr = np.sqrt(np.diag(pcov))

        pworst = popt + coeff_std*perr
        pbest = popt - coeff_std*perr
  
        total_xaxis = np.array(list(range(-len(ydata) + days_past, days_future))) + 1
       
        if plt_forecast==True:
            ax.plot(date_total_xaxis[:len(total_xaxis)], curve(total_xaxis, *popt), color=colors[j-day_start])
            #ax.fill_between(date_future_axis, curve(future_axis, *pbest), curve(future_axis, *pworst), 
            #facecolor='red', alpha=0.2)
            #print('Integral=',np.trapz(curve(total_xaxis, *popt)))
    
    if daily:
        sm = plt.cm.ScalarMappable(cmap='YlGn', norm=plt.Normalize(vmin=day_start+1, vmax=len(ydata1)+1))
    else:
        sm = plt.cm.ScalarMappable(cmap='YlGn', norm=plt.Normalize(vmin=day_start, vmax=len(ydata1)))

    plt.colorbar(sm,label='No. of days in data')
    
    ax.scatter(date_xdata, ydata1, color='blue', label='real data',s=8)
    ax.plot(date_xdata, ydata1, color='blue',alpha=1.0)
    ax.legend(loc='upper left')

    #plt.ylim(bottom=0, top=1.1*max(ydata1)) 
    
    fig=plt.gcf()
    plt.show()
    if do_imgs:
        fig.savefig('plots/evolution'+ title + '.png', dpi=200)


    return 

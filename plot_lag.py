import matplotlib.pyplot as plt

def plot_lag(x,y,last_date,country,savefig=False,string=' Cross-correlation: Daily infection - Daily death'):


    ## The position of the peak shows the lag between the two time series
    fig, ax = plt.subplots(figsize=(10,6))
    
    plt.xcorr(x,y,maxlags=None,color='blue',usevlines=True,alpha=1,normed=False)
    plt.axvline(0,color='r',alpha=0.5)
  
    plt.xlabel('Lag [days]')
    plt.ylabel(string)

    plt.title(country+': '+str(last_date.strftime("%d-%m-%Y")))
    #plt.legend(loc=2)
    plt.savefig('plots/'+str(country)+'_lag.png') #+(str(country)
    plt.show()

    return


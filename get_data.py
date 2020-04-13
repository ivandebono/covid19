"""
Ivan Debono
April 2020

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def select_country(dataframe,dataframe_r,dataframe_d,country=None,region=None):

    if country:
        dataframe=dataframe[dataframe['Country/Region']==country]
      #  dataframe=dataframe[dataframe['Province/State'].isnull()]

        dataframe_r=dataframe_r[dataframe_r['Country/Region']==country]
       # dataframe_r=dataframe_r[dataframe_r['Province/State'].isnull()]

        dataframe_d=dataframe_d[dataframe_d['Country/Region']==country]
        #dataframe_d=dataframe_d[dataframe_d['Province/State'].isnull()]

        if country == 'China':
            dataframe=dataframe.groupby(by=[('Country/Region')],as_index=False).sum()
            dataframe_r=dataframe_r.groupby(by=[('Country/Region')],as_index=False).sum()
            dataframe_d=dataframe_d.groupby(by=[('Country/Region')],as_index=False).sum()
    else:
        country = 'Global'

    return dataframe,dataframe_r,dataframe_d,country

def select_region(dataframe,dataframe_r,dataframe_d,region=None):

    if region:
        dataframe=dataframe[dataframe['Province/State'] == region ]
        dataframe_r=dataframe_r[dataframe_r['Province/State'] == region]
        dataframe_d=dataframe_d[dataframe_d['Province/State'] == region]
    else:
        dataframe=dataframe[dataframe['Province/State'].isnull()]
        dataframe_r=dataframe_r[dataframe_r['Province/State'].isnull()]
        dataframe_d=dataframe_d[dataframe_d['Province/State'].isnull()]
    
    return dataframe,dataframe_r,dataframe_d

def clean_dataframe(dataframe):



    if dataframe['Country/Region'].unique().all()=='China':
        df=dataframe.drop(columns=['Country/Region', 'Lat', 'Long'])
    else:
        df=dataframe.drop(columns=['Province/State', 'Country/Region', 'Lat', 'Long'])
    # Transpose df so date columns are rows
    df=df.transpose()
    # Get totals
    #df['Total']=df.sum(axis=1)
    # Convert index column (dates) to a column

    df.reset_index(inplace=True)
    df.rename(columns={"index": "Date"},inplace=True)
    df.Date=pd.to_datetime(df.Date, infer_datetime_format=True)  
    Total=df.drop(columns='Date').sum(axis=1)
    df['Total']=Total

    return df



def parse_data(df,df_r,df_d):
# Parse data and compute time series -----------------------

    date_string = df.iloc[-1:]['Date'].values[0]
    date_format = "%Y-%m-%dT%H:%M:%S"
    last_date = pd.Timestamp(date_string).to_pydatetime()
    print("Last update: {}".format(last_date))

    totalinfected = np.array(df['Total'].tolist())
    dailyinfected = totalinfected[1:] - totalinfected[:-1]
    print('Total infected: {}'.format(totalinfected[-1]))
    print('Total infected today: {}'.format(dailyinfected[-1]))
    
    totaldead = np.array(df_d['Total'].tolist())
    dailydead = totaldead[1:] - totaldead[:-1]
    print('Total dead: {}'.format(totaldead[-1]))
    print('Total dead today: {}'.format(dailydead[-1]))


    totalrecovered = np.array(df_r['Total'].tolist())
    dailyrecovered = totalrecovered[1:] - totalrecovered[:-1]
    print('Total recovered: {}'.format(totalrecovered[-1]))
    print('Total new recovered today : {}'.format(dailyrecovered[-1]))


    # Growth factor
    gf_list = dailyinfected[1:] / dailyinfected[:-1]
    growth_factor = gf_list[-1]
    print('Growth factor: {:.3f}'.format(growth_factor))

    avg_growth_factor = np.mean(gf_list[-3:])
    print('Mean growth factor: {:.3f}'.format(avg_growth_factor))

    #print(gf_list)
    
    return totalinfected,dailyinfected,totaldead,dailydead,totalrecovered,dailyrecovered,last_date


def plot_comparison(df,df_r,df_d,country,last_date,savefig=False):
    #%matplotlib notebook
    fig,ax=plt.subplots(figsize=(10,6))
    active=df.Total.subtract(df_r.Total.add(df_d.Total))
    #df.plot(x='Date',y='Total',legend=False)
    plt.plot(df_r.Date,df_r.Total,label='Recoveries',color='grey')
    #plt.plot(df.Date,active,label='Active cases',color='r')

    plt.plot(df_d.Date,df_d.Total,label='Deaths',color='black')

    plt.plot(df.Date,df.Total,label='Confirmed infections',color='blue')
    plt.title(str(country)+': ' + str(last_date.strftime("%d-%m-%Y")))
    plt.ylabel('Total')
    plt.tight_layout()
    plt.legend()
    if savefig:
        plt.savefig('plots/'+str(country)+'Comparison.png')
    
    return 


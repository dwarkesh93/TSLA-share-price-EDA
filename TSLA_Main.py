# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 16:50:04 2021

@author: Dwarkesh
"""

#initialization block
import pandas as pd
import numpy as np
#import scikit-learn
from sklearn.linear_model import LinearRegression, LogisticRegression
import yahoo_fin.stock_info  as yf
from matplotlib import pyplot as plt
import seaborn as sn
from pandas.plotting import lag_plot
import datetime
#data source range from 15-01-2016 to 14-01-2021
#http://theautomatic.net/2020/05/05/how-to-download-fundamentals-data-with-python/

start_date=datetime.datetime(2020,1,2)

#time series of TSLA
tsla=pd.read_csv("C:\\Users\\Dwarkesh\\Documents\\UC\\Spring\\BANA7051- Stats Methods\\Project 1\\TSLA.csv")
gm=pd.read_csv("C:\\Users\\Dwarkesh\\Documents\\UC\\Spring\\BANA7051- Stats Methods\\Project 1\\GM.csv")
dji=pd.read_csv("C:\\Users\\Dwarkesh\\Documents\\UC\\Spring\\BANA7051- Stats Methods\\Project 1\\DJI.csv")

dji.iloc[:,4]
tsla['Date']=pd.to_datetime(tsla['Date'])
gm['Date']=pd.to_datetime(gm['Date'])
dji['Date']=pd.to_datetime(dji['Date'])
temp_tsla=tsla[tsla.Date>= start_date]
temp_gm=gm[gm.Date>= start_date]
temp_dji=dji[dji.Date>= start_date]
temp_dji=temp_dji.rename(columns={" Open":"Open"," High":"High"," Low":"Low"," Close":"Close"})
temp_tsla.columns
temp_gm.columns
temp_dji.columns
#temp_dji2.columns

temp_tsla=temp_tsla.sort_values(by='Date')
temp_gm=temp_gm.sort_values(by='Date')
temp_dji=temp_dji.sort_values(by='Date')
#%%
#method definition block

def show_2dplot(b,j,i,title,df1=temp_gm,df2=temp_dji):
    plt.plot(b.iloc[:,j],b.iloc[:,i])
    plt.plot(df1.iloc[:,j],df1.iloc[:,i])
    #plt.plot(df2.iloc[:,j],df2.iloc[:,i-1])
    plt.title(title)
    plt.show()
def cum_percent_change(df):
    #adding columns to stock_price data frame and calculate absolute cumulative difference
    day1=2
    #print(df[df['Date']==start_date]['Close'])
    day1=int(df[df['Date']==start_date]['Close'])
    df['cum_percent_change']=100*(df['Close']-day1)/day1

def daily_percent_change(df):
    #adding columns to stock_price data frame and calculate relative cumulative difference compared to previous day
    df['daily_percent_change']=100*(df['Close'].diff())/df['Close']
    #print(df[['daily_percent_change','Close']])
    

def plot_lag(df,lag):
    #plt.figure(figsize=(10,10))
    lag_plot(df['Close'], lag=lag)
    plt.title(str(lag) + "-day lag scatterplot of Tesla's stock price")
    plt.show()
    
def slr_tsla(np_x,np_y,x_title,y_title):
    lm_model=LinearRegression()
    lm_model.fit((np_x),np_y)
    r2=lm_model.score(np_x,np_y)
    print("Rsquare value:: ",r2)
    show_2dplot(np_x,lm_model.intercept_+lm_model.coef_*np_x,"lag 1")
    plt.plot(np_x,lm_model.intercept_+lm_model.coef_*np_x)
    plt.plot(np_x,np_y,'o')
    plt.xlabel(x_title)
    plt.ylabel("Closing price")
    plt.title(label="Rsquare:: "+str(r2))
    plt.show()
#show_2dplot(np_id2,np_y,"lag 1")
#show_2dplot(np_vol,np_y,"lag 1")

def log_lr_tsla(np_x,np_y,x_title,y_title):
    lm_model=LogisticRegression(solver='liblinear',random_state=0)
    lm_model.fit((np_x),np_y)
    r2=lm_model.score(np_x,np_y)
    print("Rsquare value:: ",r2)
    print("intercept:: ",lm_model.intercept_)
    print("coeff::  ",lm_model.coef_)

    #show_2dplot(np_x,lm_model.intercept_+lm_model.coef_*np_x,"lag 1")
    plt.plot(np_x,lm_model.intercept_+lm_model.coef_*np_x)
    plt.plot(np_x,np_y,'o')
    plt.xlabel(x_title)
    plt.ylabel("Up or Down")
    plt.title(label="Rsquare:: "+str(r2))
    plt.show()
    
    #plt.plot(np_x,np_y,'o')
    #plt.plot(np_x,lm_model.predict(np_x),'o')
    #plt.show()


    
#%%

#GM code
    
#time series of GM
#print(dji[dji.Date>=start_date].head())

#plt.plot(gm[gm.Date>=start_date].Date,gm[gm.Date>=start_date].Open)
#plt.title('GM')

#plt.show()
tsla.head()

tsla.describe()
#musk_tweets=pd.read_csv("C:\\Users\\Dwarkesh\\Documents\\UC\\Spring\\BANA7051- Stats Methods\\Project 1\\elonmusk_tweets.csv")
#print(musk_tweets.tail(5))
#%%
    
#show_2dplot(tsla[tsla['Date']>=start_date]['Date'],tsla[tsla.Date>=start_date]['Open'],'TSLA')   
#show_2dplot(tsla[tsla['Date']>=start_date]['Date'],tsla[tsla.Date>=start_date]['cum_percent_change'],'TSLA % abs change')   
daily_percent_change(tsla)

print(tsla[['Date','daily_percent_change']])
#cum_percent_change(gm)
#trying out yahoo finance APIs
#financial API code
print(yf.get_income_statement('GM', yearly=False))
print(yf.get_income_statement('TSLA', yearly = False))



#show_2dplot(temp_tsla.Volume,temp_tsla.Close,"trend")

#%% Statistical summary

tsla.boxplot(column=['Open', 'High', 'Low', 'Close', 'Adj Close'])

summary_stat=tsla.describe()[1:]
summary_stat
summary_stat.to_csv("C:\\Users\\Dwarkesh\\Documents\\UC\\Spring\\BANA7031- Probability\\Tesla_Stat_Summary.csv")

#%% Plotting and visualization

#valuation analysis and visulization block
#print(tsla.Date)
#print(tsla[tsla['Date'==(start_date)]])

daily_percent_change(temp_tsla)
cum_percent_change(temp_tsla)
daily_percent_change(temp_gm)
cum_percent_change(temp_gm)
daily_percent_change(temp_dji)
cum_percent_change(temp_dji)
temp_tsla.head()
temp_gm.head()
temp_dji.head()

plt.plot(temp_tsla.Date,temp_tsla.daily_percent_change)
plt.title("Daily percentage change in Tesla's stock price")
plt.show()
temp_tsla.columns
show_2dplot(temp_tsla,0,5,'Tesla vs GM Closing stock price')   
show_2dplot(temp_tsla,0,6,'Tesla vs GM volume of shares traded')   
show_2dplot(temp_tsla,0,7,'Tesla vs GM % relative daily change in stock price')   
show_2dplot(temp_tsla,0,8,'Tesla vs GM % absolute change in stock price')   
#show_2dplot(temp_tsla,0,2,'TSLA % absolute change in stock price')   





#show_2dplot(temp_tsla.Date,temp_tsla.cum_percent_change,'TSLA % absolute change in stock price')   
#show_2dplot(temp_tsla.Date,temp_tsla.daily_percent_change,'TSLA % relative change in stock price(daily)')   
plt.figure(figsize=(10,10))
lp=lag_plot(temp_tsla['Close'], lag=30)
lp=lag_plot(temp_tsla['Close'], lag=14)
lp=lag_plot(temp_tsla['Close'], lag=6)

plot_lag(temp_tsla, 2)
plot_lag(temp_tsla, 5)
plot_lag(temp_tsla, 12)
plot_lag(temp_tsla, 30)
    
lp
plt.figure(figsize=(10,10))
lp=lag_plot(temp_tsla['Close'], lag=30)
plt.title("Daily percentage change in Tesla's stock price")
plt.show()

#%%
#tweet analysis block
file_addr="C:\\Users\\Dwarkesh\\Documents\\UC\\Spring\BANA7051- Stats Methods\\Project 1\\new_elonmusk_tweets.csv"
elon_tweet_dump=pd.read_csv(file_addr)
elon_tweet_dump.rename({'created_at':'Date'},inplace=True,axis=1)
print(elon_tweet_dump.columns)

#grouping by date and aggregating over count
elon_tweet_dump['Date']=pd.to_datetime(elon_tweet_dump['Date']).dt.date

#print(elon_tweet_dump['Date'].dt.date)

tweet_freq=elon_tweet_dump.groupby(by="Date")['id'].count()
tweet_freq.head()
#tweet_freq.rename(columns={"Date":"Date2"},inplace=True)
#%%

#%%
tweet_freq.to_csv('C:\\Users\\Dwarkesh\\Documents\\UC\\Spring\\BANA7051- Stats Methods\\Project 1\\OP\\tweet_freq.csv')
temp_tsla.to_csv('C:\\Users\\Dwarkesh\\Documents\\UC\\Spring\\BANA7051- Stats Methods\\Project 1\\OP\\temp_tsla.csv')


#convert tweet date into Date object
df_tweet=pd.DataFrame(tweet_freq)
df_tweet=df_tweet.reset_index()
#df_tweet.drop('Date',axis=1)
df_tweet.head()

df_tweet['Date']=df_tweet.Date.astype('str')
df_tweet=df_tweet.set_index('Date')
df_tweet.index
#df_tweet['Date']=df_tweet.Date.astype('datetime64[ns]')
#df_tweet['Date']=pd.to_datetime(df_tweet['Date']).dt.date
#df_ind=df_tweet.index

temp_tsla.head()
temp_tsla.reset_index()

#df_tweet['Date2']=df_tweet['Date']
temp_tsla['Date2']=temp_tsla['Date']
df_tweet.info()
temp_tsla.info()
temp_tsla.columns
df_tweet.columns
temp_tsla.info()

temp_tsla.Date2=temp_tsla.Date2.astype('str')

#df_tweet.index=df_tweet.astype('str')


join_data=temp_tsla.join(df_tweet,on='Date2')
#join_data=temp_tsla.join(df_tweet,lsuffix='ts', rsuffix='tw',on='Date2')

join_data.head()
print(join_data.info())
print((join_data[join_data['id']>0]))
tsla_tweet=join_data[join_data['id']>0]

tsla_tweet['Up']=(tsla_tweet['daily_percent_change']>=0).astype(int)
#tsla_tweet['Up']=sum(tsla_tweet[tsla_tweet['daily_percent_change']>0])
tsla_tweet.head()
tsla_tweet.info()
#%% correlation heatmap
corr_tsla_og=temp_tsla.reset_index().corr()

corr_tsla=tsla_tweet.corr()
sn.heatmap(corr_tsla, annot=True)
tsla_tweet_corr=tsla_tweet[['Close','Volume','daily_percent_change','id','Up']].corr()

sn.heatmap(tsla_tweet_corr,annot=True)


sn.heatmap(corr_tsla, annot=True)
plt.show()
#%% Having fun with lead /lagging indicators
lag12=pd.concat([tsla_tweet,tsla_tweet.id.shift().rename("id1"),tsla_tweet.id.shift(2).rename("id2")],axis=1)
corr_tsla=lag12[['Close','Volume','daily_percent_change','id','Up','id1','id2']].corr()
sn.heatmap(corr_tsla, annot=True)

lag12.head()
lag12=lag12.dropna()
lag12.info()

#%%
print(tsla.head())
tsla.to_csv('C:\\Users\\Dwarkesh\\Documents\\UC\\Spring\\IS6030- Data Management\\HW5\\TSLA_ext.csv')

#%% working on regressions
np_close=np.array(lag12.Close)
np_id=np.array(lag12.id)
np_id1=np.array(lag12.id1)
np_id2=np.array(lag12.id2)
np_vol=np.array(lag12.Volume)
np_up=np.array(lag12.Up)


#np_x=np.stack((np_vol).reshape(-1,1))
np_x=np_vol.reshape(-1,1)
np_x
(np_close)

np_y=np_up

lag12.describe()[1:]
quantile_val=lag12.iloc[:,-4].quantile(.95)
lag_trim=lag12[lag12.iloc[:,-4]<quantile_val]
#%%
#lag trim computation

lag_trim_id=np.array(lag_trim.id) 
lag_trim_up=np.array(lag_trim.Up)
  

#%%

log_lr_tsla(np.stack(lag_trim_id.reshape(-1,1)),lag_trim_up, "Tweet # today", "Up")  
 
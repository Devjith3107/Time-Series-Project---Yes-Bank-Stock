#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import scipy.stats
import pylab
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.graphics.tsaplots as sgt
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
import statsmodels.api as sm
from scipy.stats.distributions import chi2
from statsmodels.tsa.statespace.sarimax import SARIMAX
import yfinance
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.api import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from arch import arch_model


# In[2]:


from pmdarima import auto_arima


# In[3]:


Yes=pd.read_csv("YesBank_StockPrices.csv")


# In[4]:


Yes.head()


# In[5]:


Yes


# In[6]:


Yes.head(10)


# In[7]:


Yes.tail(10)


# In[8]:


Yes.info()


# In[9]:


Yes.describe()


# In[10]:


Yes.tail(10)


# ##### Checking for missing values

# In[11]:


Yes.isna().sum()


# #### As one can see there is no missing values for our dataframe respectively

# ##### Creating return table where return is Closing Price minus opening price divided by the opening Prices respectively.

# In[12]:


Yes['return']=(Yes['Close']-Yes['Open']).div(Yes['Open']).mul(100)


# In[13]:


Yes['return'].plot(x=Yes["Date"])


# In[14]:


Yes


# #### Preparing the Time Series and Basic EDA

# To be able to apply ARIMA to a data, the date column needs to be converted into a date time object and then made the index of the dataframe. This is achieved by using strptime of the datetime library. The Given Date format MMM-YY is converted to proper date of YYYY-MM-DD, that Date is set as index and frequency of the Date is set to 'MS' which is monthly

# In[15]:


from datetime import datetime
Yes['Date'] = Yes['Date'].apply(lambda x: datetime.strptime(x, '%b-%y'))


# In[16]:


Yes.head()


# #### As one can see the Month Year format that was in words earlier have been converted to proper Year Month format respectively

# #### Dropping the index column and using the Date column instead of it respectively

# In[17]:


Yes.index=Yes['Date']


# In[18]:


Yes.drop('Date',axis=1,inplace=True)


# In[19]:


Yes


# In[20]:


Yes.index.freq = 'MS'


# In[21]:


Yes


# ### EDA (Exploratory Data Analysis)

# In[22]:


Yes.plot()


# #### Though seasonality is not evident from the plotting, one can see other than the return plot, all the Opening Price, Closing Price, Highest Price, Lowest Price all follows an upward sloping trend till 2018 and then the sharp downward sloping pattern after 2018, this thing can be attributed to the Yes bank detoriating financial condition that started from 2018.Companies that failed to repay their loans to YES Bank include Dewan Housing Finance, Essel group, CG Power, the Anil Ambani group companies, and Videocon.The crisis at the bank began in 2018 and grew gradually. In September that year, the RBI reduced Rana Kapoor’s new three-year term as CEO until January 31, 2019. The next day, the YES Bank stock tanked 30 per cent and continued its downward spiral. For complete details more can be read at https://www.business-standard.com/podcast/companies/yes-bank-stock-crash-and-financial-mess-how-the-crisis-unfolded-in-2-years-120030600886_1.html respectively.

# For our analysis we will mainly use the Closing Price column and return column mainly 

# Now that the dataframe is ready, we divide it into train and test for modeling and testing. For this example, The last two years, Jan 2019- Nov 2020, are taken as test, Rest everything is train, Since we have 185 observations and complete monthly data for each year, last 2 years account to 24 months and and hence 161 (185-24) observations are kept for training and rest is for testing respectively.

# In[23]:


#### Train test split
data = Yes.iloc[:162]
test = Yes.iloc[162:]


# In[24]:


#### Plotting the data
data[['Close','return']].plot()


# #### Checking for stationarity with ADF test

# Checking for stationarity of Closing Price column

# In[25]:


adfuller(data['Close'])


# #### Cleary Closing price is non stationary as we fail reject The Null Hypothesis of Non Stationarity respectively, Hence to have a clear idea we check the acf plot and seasonal decomposition of Closing Price respectively

# #### The ACF plot for closing price

# In[26]:


sgt.plot_acf(data['Close'],lags=50)


# In[27]:


sgt.plot_pacf(data['Close'],lags=80,method='ols')


# ##### From the plot of acf above it is clear for any time period, the Closing price has a high correlation with almost 12-14 lags or for time t yt is infuenced by all it's previous or lagged values upto almost y(t-14) respectively which implies the data might be non-stationary hence any model incorporated should have higher order lags for better prediction respectively, the same pacf plot shows that every lag is significant , also from acf plot we find strong evidence of theoritical ARMA model acf respectively

# In[28]:


SD_Close_add=seasonal_decompose(data['Close'],model='additive')


# In[29]:


SD_Close_add.plot()


# In[30]:


SD_Close_mul=seasonal_decompose(data['Close'],model='multiplicative')


# In[31]:


SD_Close_mul.plot()


# #### Clearly Closing price demonstrates both positive average trend and seasonality that implies one might use SARIMAX models over ARIMA, AR, or MA models for prediction respectively

# In[32]:


data.groupby(data.index.year).mean().Close.plot()


# ##### Finally from the plot of yearwise average it is quite evident that average is rising with few dips and dumbs and falling drasticallty after 2018 but still positive implies overall average will be still be away 0, We finally calculate the mean of Closing price column respectively

# In[33]:


data['Close'].mean()


# Checking for stationarity of Return column

# In[34]:


data['return'].plot()
plt.axhline(y=0, color='r', linestyle='-', label='y=0')
#Here instead of plt.plot ,for plotting the horizontal line we use axhline or axis horizontal line respectively


# #### Note clearly returns don't have a predictable trend which implies returns are not non-stationary, more random and might be uncorrelated over lag periods respectively

# In[35]:


adfuller(data['return'])


# #### Clearly from the data above we reject the null hypothesis of non stationarity respectively which implies our returns is stationary over time respectively

# #### The ACF and PACF plot of return columns 

# In[36]:


sgt.plot_acf(data['return'],lags=100,zero=False)


# #### Clearly from the plot above we find past lags of any order are insignificant in terms of their correlation with their present value terms respectively which sends another indication why data is stationary respectively

# In[37]:


sgt.plot_pacf(data['return'],lags=80,zero=False,method="ols")


# #### However from the pacf plot there does exist some significant correlation with the past values. Finally we check for seasonal decomposition

# In[38]:


SD_Return_add= seasonal_decompose(data['return'],model="additive")


# In[39]:


SD_Return_add.plot()


# ##### Note the trend is more or less around 0, so the data is stationary

# ### Conversion to stationary dataframe

# #### Applying difference by difference method

# In[40]:


data['Stat_Close']=data['Close'].diff(1)


# In[41]:


adfuller(data['Stat_Close'][1:])


# ##### Note after first order difference data is still not stationary , so we will use second order differencing to convert our data into stationarity

# In[42]:


data['Stat_Close2']=data['Stat_Close'].diff(1)


# In[43]:


adfuller(data['Stat_Close2'][2:])


# #### Clearly our data is now stationary and we can use the second order difference equation for our time Series Prediction

# In[44]:


data['Stat_Close2'].plot()


# #### Log differencing

# In[45]:


data['Log_Close']=np.log(data['Close']).diff(1)


# In[46]:


adfuller(data['Log_Close'][1:])


# In[47]:


data['Log_Close'][1:].plot()


# #### Hence with Log differencing our data is stationary for every level of significance however we lose 1 period of observation

# In[48]:


data.shape


# In[49]:


# Using statmodels: Subtracting the Trend Component.
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
res = seasonal_decompose(data['Close'], extrapolate_trend='freq')
detrended = data.Close.values - res.trend
plt.plot(detrended)
plt.title('Stock Prices detrended by subtracting the trend component', fontsize=16)


# In[50]:


adfuller(detrended)


# In[51]:


data['detrended']=pd.DataFrame(detrended)


# ##### With linear detrending our data though significantly stationary at 5% level of significance is not that significance at 1% one, however the advantage of time Series differencing over Log differencing is with time series differencing, we don't lose out the first observation respectively 

# In[52]:


res.trend.plot()
data.Close.plot()


# In[53]:


SD_Close_add.trend.plot()
res.trend.plot()


# ### Prediction

# #### Here we do 2 predictions
# ##### 1)Till 2017 data will be collected and we will use it to predict the stock prices from 2018-2019 and 2019-2020
# ##### 2)Till 2018, data will be collected and and predictions of 2019-2020 will be done in order to predict 2019-2020 and difference in predictions will be taken into account for capturing the distortion due to scam respectively.
# ##### 3)Second prediction series will be compared with actual predictions of 2019-2020 respectively  

# In[54]:


#### Creating the data with and without scam
scam_date="2018-01-01"
Prescam=data[:"2017-12-01"]
PostScam=data[scam_date:]


# #### Before predicting Post Scam Date we use first find the best model of from training data set by creating a sub train test split in our Training data respectively

# In[55]:


Prescam.shape


# In[56]:


# Set one year for testing
strain = detrended.iloc[:138]
stest = detrended.iloc[138:150]


# In[57]:


strain


# In[58]:


stest


# ### Checking the acf and pacf plot to determine the optimum order of lag

# In[59]:


sgt.plot_acf(Prescam['detrended'])


# In[60]:


sgt.plot_pacf(Prescam['detrended'],method="ols",lags=50)


# #### From the ACF plot it is quiet evident that lags till order 4 are quite significant so we start with an initial guess of order 4 of MA process, however with PACF plot almost all lags is significant however due to risk of severe overfitting , we might just start with MA(4) and AR(4) model 

# ### Prediction using ARMA(4,4) model

# In[61]:


ARMA44=ARIMA(Prescam['detrended'],order=(4,0,4))


# In[62]:


ar44=ARMA44.fit()


# In[63]:


ar44.summary()


# In[64]:


AR44_predict=ar44.predict(start="2017-01-01",end="2017-12-01")


# In[65]:


fig,axes=plt.subplots(figsize=(10,5))
AR44_predict.plot()
stest.plot()
plt.show()


# In[66]:


for i in range(0,len(AR44_predict)):
    print(f"predicted={AR44_predict[i]}, expected={stest[i]}")


# In[67]:


error = mean_squared_error(stest,AR44_predict)
print(f'ARMA(4,4) MSE Error: {error:11.10}')


# In[68]:


Error_AR44=stest-AR44_predict


# In[69]:


sgt.plot_acf(Error_AR44)


# In[70]:


sgt.plot_pacf(Error_AR44,lags=5)


# ### Though the acf plot shows white noise errors using both pacf and acf plots respectively

# #### Note all ar4, and all ma terms have been found to be insignificant so we revise our model with ar 3 terms 

# #### AR(3)

# In[71]:


AR3=ARIMA(Prescam['detrended'],order=(5,0,0))


# In[72]:


ar3=AR3.fit()


# In[73]:


ar3.summary()


# In[74]:


AR3_predict=ar3.predict(start="2017-01-01",end="2017-12-01")


# In[75]:


fig,axes=plt.subplots(figsize=(10,5))
AR3_predict.plot()
stest.plot()
plt.show()


# In[76]:


for i in range(0,len(AR3_predict)):
    print(f"predicted={AR3_predict[i]}, expected={stest[i]}")


# In[77]:


error = mean_squared_error(stest,AR3_predict)
print(f'ARMA(3) MSE Error: {error:11.10}')


# In[78]:


Error_AR3=stest-AR3_predict


# In[79]:


sgt.plot_acf(Error_AR3)


# In[80]:


sgt.plot_pacf(Error_AR3,lags=5)


# In[ ]:





# In[ ]:





# #### SARIMA Model

# In[81]:


Sarima=auto_arima(Prescam['detrended'],
    start_p=2,
    d=2,
    start_q=2,
    max_p=None,
    max_d=5,
    max_q=None,
    start_P=2,
    D=1,
    start_Q=2,
    max_P=None,
    max_D=2,
    max_Q=None,
    max_order=None,
    m=12,
    seasonal=True,
    information_criterion='aic',
    alpha=0.05,
    trend=None,
    maxiter=200,
    suppress_warnings=True)


# In[82]:


Sarima.summary()


# In[83]:


from statsmodels.tsa.statespace.sarimax import SARIMAX
  
Sarima_model = SARIMAX(Prescam['detrended'], 
                order = (7, 2, 0),seasonal_order=(1,1,0,12))
  
Sarima_result = Sarima_model.fit(maxiter=200)
Sarima_result.summary()


# In[84]:


sstart="2017-01-01"
send="2017-12-01"


# In[85]:


Prescam


# In[86]:


SARIMAPredictions=Sarima_result.predict(start=sstart,end=send)


# In[87]:


SARIMAPredictions.plot()
stest.plot()


# In[88]:


error = mean_squared_error(stest, SARIMAPredictions)
print(f'SARIMA MSE Error: {error:11.10}')


# #### As one can see using SARIMA model offers a huge improvement over normal AR(2) model as not only it takes account of seasonality and higher order error lags that is not only includes the MA part but also each coefficient has been significant respectively

# In[89]:


WN_Error=SARIMAPredictions-stest


# In[90]:


WN_Error.plot()


# In[91]:


adfuller(WN_Error)


# In[92]:


sgt.plot_acf(WN_Error)


# In[93]:


sgt.plot_pacf(WN_Error,method='ols',lags=5)


# ### With Exogenous factors

# In[94]:


Auto_Arima=auto_arima(Prescam['detrended'],X=Prescam[['Open','High','Low']],
    start_p=2,
    d=2,
    start_q=2,
    max_p=None,
    max_d=5,
    max_q=None,
    start_P=1,
    D=1,
    start_Q=1,
    max_P=None,
    max_D=2,
    max_Q=None,
    max_order=None,
    m=12,
    seasonal=True,
    information_criterion='aic',
    alpha=0.05,
    trend=None,
    maxiter=200,
    suppress_warnings=True)


# In[95]:


Auto_Arima.summary()


# In[96]:


from statsmodels.tsa.statespace.sarimax import SARIMAX
  
model = SARIMAX(Prescam['detrended'],exog=Prescam[['Open','High','Low']], 
                order = (1, 2, 2),seasonal_order=(1,1,1,12))
  
result = model.fit(maxiter=200)
result.summary()


# In[97]:


Predictions=result.predict(start=sstart,end=send)


# In[98]:


stest.plot(title="Predicted vs Actual Detrended Closing Prices")
Predictions.plot()
plt.legend(["Predicted Closing Price FY 2017","Actual Closing Prices FY 2017"],loc = "upper left")


# In[99]:


error = mean_squared_error(stest, Predictions)
print(f'SARIMAX MSE Error: {error:11.10}')


# #### As one can see we have not only a good prediction but also have a very low RMSE error as well, however this might imply some cause of overfitting as well

# In[100]:


res.trend


# In[101]:


res.trend.plot(title="Time Trend of Closing Prices")


# In[102]:


trendy = pd.DataFrame(res.trend)[:"2017-12-01"]


# In[103]:


trendy.plot()


# In[104]:


# Set one year for testing
traint = trendy.iloc[:138]
testt = trendy.iloc[138:150]


# In[105]:


Trend_Arima=auto_arima(trendy,
    start_p=2,
    d=2,
    start_q=2,
    max_p=None,
    max_d=5,
    max_q=None,
    start_P=1,
    D=1,
    start_Q=1,
    max_P=None,
    max_D=2,
    max_Q=None,
    max_order=None,
    seasonal=False,
    information_criterion='aic',
    alpha=0.05,
    trend=None,
    maxiter=200,
    suppress_warnings=True)


# In[106]:


Trend_Arima.summary()


# In[107]:


modelt = SARIMAX(trendy['trend'],order=(4,2,3))
resultst = modelt.fit(maxiter=200)
resultst.summary()


# In[108]:


# Obtain predicted values
start=len(traint)
end=len(traint)+len(testt)-1
predictionst = resultst.predict(start=start, end=end, dynamic=False, typ='levels', full_results = True).rename('SARIMA Predictions')


# In[109]:


# Compare predictions to expected values
for i in range(len(predictionst)):
    print(f"predicted={predictionst[i]:<11.10}, expected={testt['trend'][i]}")


# In[110]:


ax = testt['trend'].plot(legend=True,figsize=(6,6))
predictionst.plot(legend=True)
ax.autoscale(axis='x',tight=True)


# In[111]:


from sklearn.metrics import mean_squared_error

error = mean_squared_error(testt['trend'], predictionst)
print(f'SARIMA MSE Error: {error:11.10}')


# #### Clearly using SARIMAX model we can predict trend with a very little RSE , we could have used linear regression model but owing to the concavity and convexity trend displays over ranges of time , SARIMAX did a better prediction. Once both trend component and seasonality + random compoenent were predicted for 2017 They were added together and compared with the original 2017 values

# In[112]:


finalpreds = (predictionst + Predictions)


# In[113]:


data["2017-01-01":"2017-12-01"]['Close'].plot()
finalpreds.plot()


# In[114]:


from sklearn.metrics import mean_squared_error

error = mean_squared_error(data["2017-01-01":"2017-12-01"]['Close'], finalpreds)
print(f'SARIMA MSE Error: {error:11.10}')


# #### We now use our previous SARIMA model to predict 2018 data

# #### Since till 2017 data we did not have a scam and hence we use this data to forecast stock prices from year 2018 which we call as prediction without scam

# In[115]:


Predictions_NoScam=Sarima_result.forecast(steps=36)


# In[116]:


Predictions_NoScam.plot()


# ##### Predictions for time component from 2018 onwards

# In[117]:


resultst_NoScam=resultst.forecast(steps=36, dynamic=False, typ='levels', full_results = True)


# In[118]:


No_Scam=Predictions_NoScam+resultst_NoScam


# In[119]:


Yes['Close']['2018-01-01':].plot()
No_Scam.plot()


# In[120]:


Unantipated=No_Scam-Yes['Close']['2018-01-01':]


# In[121]:


Unantipated.plot()


# ##### Fianlly we are going to predict the same incorporating 2018 into effect 

# In[122]:


Sarima_Scam=auto_arima(data['detrended'],
    start_p=2,
    d=2,
    start_q=2,
    max_p=None,
    max_d=5,
    max_q=None,
    start_P=2,
    D=1,
    start_Q=2,
    max_P=None,
    max_D=2,
    max_Q=None,
    max_order=None,
    m=12,
    seasonal=True,
    information_criterion='aic',
    alpha=0.05,
    trend=None,
    maxiter=200,
    suppress_warnings=True)


# In[123]:


Sarima_Scam.summary()


# In[124]:


from statsmodels.tsa.statespace.sarimax import SARIMAX
  
Sarima_Scam_model = SARIMAX(data['detrended'], 
                order = (6, 2, 1),seasonal_order=(1,1,0,12))
  
Sarima_Scam_result = Sarima_Scam_model.fit(maxiter=200)
Sarima_Scam_result.summary()


# In[125]:


Predictions_Scam=Sarima_Scam_result.forecast(steps=24)


# In[126]:


Predictions_Scam


# In[127]:


Predictions_Scam.plot()


# In[128]:


SD_Scam=seasonal_decompose(data['Close'],extrapolate_trend='freq')


# In[129]:


SD_Scam_trend=SD_Scam.trend


# In[130]:


SD_Scam_trend.plot()


# In[131]:


Trend_Scam_Arima=auto_arima(SD_Scam_trend,
    start_p=2,
    d=2,
    start_q=2,
    max_p=None,
    max_d=5,
    max_q=None,
    start_P=1,
    D=1,
    start_Q=1,
    max_P=None,
    max_D=2,
    max_Q=None,
    max_order=None,
    seasonal=False,
    information_criterion='aic',
    alpha=0.05,
    trend=None,
    maxiter=200,
    suppress_warnings=True)


# In[132]:


Trend_Scam_Arima.summary()


# In[133]:


modeltScam = SARIMAX(SD_Scam_trend,order=(1,2,2))
Scam_resultst = modeltScam.fit(maxiter=200)
Scam_resultst.summary()


# In[134]:


resultst_Scam=Scam_resultst.forecast(steps=24, dynamic=False, typ='levels', full_results = True)


# In[135]:


Scam_pred=Predictions_Scam+resultst_Scam


# In[136]:


Scam_pred


# In[137]:


fig,axes=plt.subplots(figsize=(20,10))
Scam_pred.plot()
No_Scam["2019-01-01":].plot()
Yes['Close']["2019-01-01":].plot()
plt.legend(["Predicted Closing Price FY 2019 with scam","Predicted Closing Price FY 2019-20 without scam","Actual Closing Prices FY 2019-20"],loc = "upper left")


# In[138]:


Yes['Close']["2019-01-01":]


# In[139]:


Yes['Close']["2019-01-01":]-No_Scam["2019-01-01":"2020-11-01"]


# In[140]:


(Yes['Close']["2018-01-01":]-No_Scam["2018-01-01":"2020-11-01"]).mean()


# In[141]:


(Yes['Close']["2019-01-01":]-No_Scam["2019-01-01":"2020-11-01"]).mean()


# ### Most important part upon forecasting, Stock prices on average lost 274(.approx) value taking the scam year into effect and lost Rs 376(.approx) from the predicted price respectively after the scam Happened respectively

# In[142]:


from sklearn.metrics import mean_squared_error

error = mean_squared_error(Yes['Close']["2019-01-01":], No_Scam["2019-01-01":"2020-11-01"])
error1 = mean_squared_error(Yes['Close']["2019-01-01":],Scam_pred["2019-01-01":"2020-11-01"])
print(f'SARIMA MSE Error for not taking scam: {error:11.10}')
print(f'SARIMA MSE Error for taking scam: {error1:11.10}')


# In[143]:


(Yes['Close']["2018-01-01":]-No_Scam["2018-01-01":"2020-11-01"]).div(No_Scam["2018-01-01":"2020-11-01"]).mul(100).mean()


# ## Yes Bank <font color='red'>on average lost almost 61% of the predicted non scam stock value starting from the scam year 2018 <font color='black'>all upto 2020 respectively

# In[144]:


(Yes['Close']["2019-01-01":]-No_Scam["2019-01-01":"2020-11-01"]).div(No_Scam["2019-01-01":"2020-11-01"]).mul(100).mean()


# ## Yes Bank <font color='red'>on average lost almost 82% of the predicted non scam stock value after the scam year starting from year 2019 <font color='black'>all upto 2020 respectively. Clearly this implies <font color='red'>2018 defamation dented the reputation of Yes Bank

# In[145]:


(Yes['Close']["2019-01-01":]-Scam_pred["2019-01-01":"2020-11-01"]).div(Scam_pred["2019-01-01":"2020-11-01"]).mul(100).mean()


# #### If the scam year is taken into account one can predict since Yes Bank's predicted stock price is lower now, from year 2019 they lost on average 33% value respectively 

# ### Thus without taking the scam in consideration our model was not just significantly overestimated prediction but after taking scam, predictions were predicted very much close to the right direction with  a downward sloping trend but our MSE came down by almost 93% while on average errors due to prediction was as high as 287 (approx) respectively

# In[146]:


Error_noScam=No_Scam["2019-01-01":]-Yes['Close']["2019-01-01":]


# In[147]:


Error_Scam=Scam_pred-Yes['Close']["2019-01-01":]


# In[148]:


Error_Scam.plot()
Error_noScam.plot()
plt.title(["Errors"])
plt.legend(['Predicted Errors with Scam','Predicted Errors without Scam'])


# In[149]:


Error_noScam.mean()


# In[150]:


Error_Scam.mean()


# In[151]:


Error_noScam.min()


# In[152]:


Error_Scam.min()


# #### Simple Exponential Smoothing

# In[153]:


model_Simple = SimpleExpSmoothing(Prescam["detrended"]).fit(optimized=True)
print(model_Simple.summary())


# In[154]:


Simple_Exp=model_Simple.predict(start='2017-01-01',end='2017-12-01')


# In[155]:


Simple_Exp.plot()
Prescam['detrended']["2017-01-01":].plot()


# In[156]:


from sklearn.metrics import mean_squared_error

error = mean_squared_error(Prescam['detrended']["2017-01-01":], Simple_Exp)
print(f'Simple Exponential Smoothing Error: {error:11.10}')


# #### Double Exponential Smoothing

# In[157]:


model_Double = ExponentialSmoothing(Prescam['detrended'],trend='add').fit(optimized=True)
print(model_Double.summary())


# In[158]:


Double_Exp=model_Double.predict(start='2017-01-01',end='2017-12-01')


# In[159]:


Double_Exp.plot()
Prescam['detrended']["2017-01-01":].plot()


# In[160]:


from sklearn.metrics import mean_squared_error

error = mean_squared_error(Prescam['detrended']["2017-01-01":], Double_Exp)
print(f'Double Exponential Smoothing Error: {error:11.10}')


# #### Triple Exponential Smoothing

# In[161]:


model_Triple = ExponentialSmoothing(Prescam['detrended'],trend='add',seasonal='add').fit(optimized=True)
print(model_Triple.summary())


# In[162]:


Triple_Exp=model_Triple.predict(start='2017-01-01',end='2017-12-01')


# In[163]:


Triple_Exp.plot()
Prescam['detrended']["2017-01-01":].plot()


# In[164]:


from sklearn.metrics import mean_squared_error

error = mean_squared_error(Prescam['detrended']["2017-01-01":], Triple_Exp)
print(f'Triple Exponential Smoothing Error: {error:11.10}')


# ### Predicting Returns 

# In[165]:


### Checking for stationarity of dataset
adfuller(Prescam['return'])


# #### Since returns are stationary for every level of significance we use the data directly for decompositon and from the seasonal decomposition plot done earlier we use level of seasonality as 12

# ### Prediction

# #### Here we do 2 predictions
# ##### 1)Till 2017 data will be collected and we will use it to predict the stock return of stock prices from 2018-2019 and 2019-2020
# ##### 2)Till 2018, data will be collected and and predictions of 2019-2020 will be done in order to predict 2019-2020 and difference in predictions will be taken into account for capturing the distortion due to scam respectively.
# ##### 3)Second prediction series will be compared with actual predictions of 2019-2020 respectively  

# ### Checking for acf of Returns

# In[166]:


sgt.plot_acf(Prescam['return'])


# In[167]:


sgt.plot_pacf(Prescam['return'])


# #### As it is visible Lag of order 17 might significantly affect current lag but due to risk of overfitting we only consider till Lag 5 which is significant as well just before the 17th lag, similarly from the PACF plot we don't extend till 15th lag , we only consider the 5th lag, hence we start with the ARMA(5,5) model

# In[168]:


ARMA5_Return=SARIMAX(Prescam['return'],order=(1,0,1),seasonal_order=(0,0,0,0))
ARMA5_return=ARMA5_Return.fit(maxiter=200)
ARMA5_return.summary()


# In[169]:


AR5predict=ARMA5_return.predict(start=start, end=end, dynamic=False, typ='levels', full_results = True)


# In[170]:


AR5predict.plot()
Prescam['return'][start:end].plot()
plt.legend(["Predicted Returns FY 2017","Actual Returns FY 2017"])


# In[171]:


from sklearn.metrics import mean_squared_error

error = mean_squared_error(Prescam['return']["2017-01-01":], AR5predict)
print(f'AR(5) returns error: {error:11.10}')


# In[172]:


AR5_Return_Err=Prescam['return']["2017-01-01":]-AR5predict


# In[173]:


sgt.plot_acf(AR5_Return_Err)


# In[174]:


sgt.plot_pacf(AR5_Return_Err,lags=5)


# In[175]:


Returns=auto_arima(Prescam['return'],
    start_p=2,
    d=2,
    start_q=2,
    max_p=None,
    max_d=5,
    max_q=None,
    start_P=2,
    D=1,
    start_Q=2,
    max_P=None,
    max_D=10,
    max_Q=None,
    max_order=None,
    m=12,
    seasonal=True,
    information_criterion='aic',
    alpha=0.05,
    trend=None,
    maxiter=200,
    suppress_warnings=True)


# In[176]:


Returns.summary()


# In[177]:


ARIMA_Return=SARIMAX(Prescam['return'],order=(7,2,0),seasonal_order=(6,1,0,12))
ARIMA_return=ARIMA_Return.fit(maxiter=200)
ARIMA_return.summary()


# In[178]:


returnARIMA=ARIMA_return.predict(start=start, end=end, dynamic=False, typ='levels', full_results = True)


# In[179]:


returnARIMA.plot()
Prescam['return'][start:].plot()
plt.legend(["SARIMA Predicted Returns 2017","Actual Returns FY 2017"])


# #### Note even without scam our SARIMA model wasn't able to predict the high volatily that came in returns due to the scam , this implies we need time series models that can better model volatilities due to such scams respectively that also implies one might use better models like GARCH or ARCH for that matter respectively

# In[180]:


from sklearn.metrics import mean_squared_error

error = mean_squared_error(Prescam['return']["2017-01-01":], returnARIMA)
print(f'SARIMA returns error: {error:11.10}')


# In[181]:


Err_Ret=returnARIMA-Prescam['return'][start:end]


# In[182]:


Err_Ret.plot()


# In[183]:


returnARIMA_all=ARIMA_return.predict(start="2005-07-01", end="2017-12-01", dynamic=False, typ='levels', full_results = True)


# In[184]:


returnARIMA_all.plot()
Prescam['return'].plot()
plt.legend(["SARIMA Predicted Returns 2005-17","Actual Returns FY 2005-17"])


# In[185]:


Err_Ret_all=returnARIMA_all-Prescam['return']


# In[186]:


Err_Ret_all.plot()


# In[187]:


sgt.plot_acf(Err_Ret_all)


# In[188]:


sgt.plot_pacf(Err_Ret_all)


# In[189]:


sq_Err_Ret=Err_Ret**2


# In[190]:


sgt.plot_acf(sq_Err_Ret[:"2017-11-01"])


# In[236]:


sgt.plot_pacf(sq_Err_Ret[:"2017-11-01"],method="ols",lags=4)


# In[191]:


NoScamSarimareturn=ARIMA_return.forecast(steps=36)


# In[192]:


NoScamSarimareturn.plot()
Yes['return']["2018-01-01":].plot()
plt.legend(['predictions from SARIMA model without scam','Actual Prediction with scam'])


# #### Note due to scam our SARIMA model wasn't able to predict the high volatily that came in returns due to the scam , this implies we need time series models that can better model volatilities due to such scams respectively that also implies one might use better models like GARCH or ARCH for that matter respectively

# In[193]:


data


# ### Predictions with scam

# In[194]:


Returns_Scam=auto_arima(data['return'],
    start_p=2,
    d=2,
    start_q=2,
    max_p=None,
    max_d=5,
    max_q=None,
    start_P=2,
    D=1,
    start_Q=2,
    max_P=None,
    max_D=10,
    max_Q=None,
    max_order=None,
    m=12,
    seasonal=True,
    information_criterion='aic',
    alpha=0.05,
    trend=None,
    maxiter=200,
    suppress_warnings=True)


# In[195]:


Returns_Scam.summary()


# In[196]:


SARIMA_Scam_Return=SARIMAX(data['return'],order=(8,2,0),seasonal_order=(4,1,1,12))
SARIMA_Scam_return=SARIMA_Scam_Return.fit(maxiter=200)
SARIMA_Scam_return.summary()


# In[197]:


Pred_Ret_Scam=SARIMA_Scam_return.forecast(steps=24)


# In[198]:


Pred_Ret_Scam.plot()
NoScamSarimareturn["2019-01-01":].plot()
Yes['return']["2019-01-01":].plot()
plt.legend(["Predicted Returns after Scam FY 2019-21","Predicted Returns with Scam FY 2019-21","Actual Returns with Scam FY 2019-21"])
plt.show()


# In[199]:


Error_Return_Scam=Yes['return']["2019-01-01":]-Pred_Ret_Scam
Error_Return_NoScam=Yes['return']["2019-01-01":]-NoScamSarimareturn["2019-01-01":]


# In[200]:


Error_Return_Scam.plot()
Error_Return_NoScam.plot()


# In[201]:


def f(x):
    y=x**2
    return y


# In[202]:


Sqd_Error_Return_Scam=Error_Return_Scam.apply(f)
Sqd_Error_Return_NoScam=Error_Return_NoScam.apply(f)


# In[203]:


sgt.plot_acf(Sqd_Error_Return_Scam[:"2020-11-01"])


# In[204]:


sgt.plot_pacf(Sqd_Error_Return_Scam[:"2020-11-01"],lags=10,method='ols')


# In[205]:


sgt.plot_acf(Sqd_Error_Return_NoScam[:"2020-11-01"])


# In[206]:


sgt.plot_pacf(Sqd_Error_Return_NoScam[:"2020-11-01"],lags=10,method='ols')


# #### Simple Exponential Smoothing

# In[207]:


model_Simple_returns = SimpleExpSmoothing(Prescam["return"]).fit(optimized=True)
print(model_Simple_returns.summary())


# In[208]:


Simple_Exp_return=model_Simple_returns.predict(start='2017-01-01',end='2017-12-01')


# In[209]:


Simple_Exp_return.plot()
data['return']['2017-01-01':'2017-12-01'].plot()


# In[210]:


error = mean_squared_error(Prescam['return']["2017-01-01":], Simple_Exp_return)
print(f'Simple Exponential Smoothing Error: {error:11.10}')


# #### Double Exponential Smoothing Error

# In[211]:


model_Double_return = ExponentialSmoothing(Prescam['return'],trend='add').fit(optimized=True)
print(model_Double_return.summary())


# In[212]:


Double_Exp_return=model_Double_return.predict(start='2017-01-01',end='2017-12-01')


# In[213]:


Double_Exp_return.plot()
data['return']['2017-01-01':'2017-12-01'].plot()


# In[214]:


error = mean_squared_error(Prescam['return']["2017-01-01":], Double_Exp_return)
print(f'Double Exponential Smoothing Error: {error:11.10}')


# #### Triple Exponential Smoothing

# In[215]:


model_Triple_return = ExponentialSmoothing(Prescam['return'],trend='add',seasonal='add').fit(optimized=True)
print(model_Triple_return.summary())


# In[216]:


Triple_Exp_return=model_Triple_return.predict(start='2017-01-01',end='2017-12-01')


# In[217]:


Triple_Exp_return.plot()
Prescam['return']['2017-01-01':'2017-12-01'].plot()


# In[218]:


error = mean_squared_error(Prescam['return']["2017-01-01":], Triple_Exp_return)
print(f'Triple Exponential Smoothing Error: {error:11.10}')


# ### Fitting an ARCH model

# In[219]:


Ret_square=Prescam['return'].mul(Prescam['return'])


# In[220]:


Ret_square.plot()


# In[221]:


sgt.plot_acf(Ret_square)


# In[222]:


sgt.plot_pacf(Ret_square,method='ols',lags=20)


# ### Here from the acf plot we find 5th order lag to be significant while from the pacf plot we find that 7th order lag to be significant 

# In[223]:


model_arch_1= arch_model(Prescam['return'][1:])
results_arch_1=model_arch_1.fit()
results_arch_1.summary()


# ### The Simple ARCH(7) model since ACF and PACF both has significant lags till spike 1

# In[248]:


model_arch_7= arch_model(Prescam['return'][1:],vol="GARCH",p=7,q=2)
results_arch_7=model_arch_7.fit()
results_arch_7.summary()


# In[249]:


ARCH_Predict=results_arch_7.forecast(start="2005-07-01")


# In[250]:


np.sqrt(ARCH_Predict.variance["2005-07-01":]).plot()
Prescam['return'].plot()
Prescamdev.plot()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





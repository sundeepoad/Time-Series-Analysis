## importing needed libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.colors as mcolors
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pmd


## import csv
data1 = pd.read_csv("BankDeposit.csv")




################### MONTH TO MONTH DEPOSIT BETWEEN 12/10/2021 - 07/03/2024 ##################

## getting date in right format
data1['Date'] = pd.to_datetime(data1['Date'], format='%d/%m/%Y %H:%M')
### Sorting the dataframe by date
data1.sort_values(by='Date', inplace=True)

# Grouping the data month by month and gettings sum of Deposit received
monthly = data1.groupby(data1['Date'].dt.month)['Deposit'].sum()

##  Bar plot for amounts earned month by month
plt.bar(monthly.index, monthly.values, color='orange', edgecolor='gray')
plt.xlabel("Month")
plt.ylabel("Deposited Amount (CAD)")
plt.title('Month to Month Deposited Amount')
plt.show()
##################



######## Day to Day DEPOSIT BETWEEN 12/10/2021 - 07/03/2024 ##################
daily = data1.groupby(data1['Date'].dt.day)['Deposit'].sum()
plt.bar(daily.index, daily.values, color = 'brown')
plt.xlabel("Day")
plt.ylabel("Deposited Amount (CAD)")
plt.title('Day to Day Deposited Amount')
plt.show()
####################################


####################################
###  Finding top paying Patients
cust = data1.sort_values(by= "Deposit", ascending = False)

cols = ["Date","Deposit", "LastPatient"]
print("Deposited Amounts : ",cust[cols].head(10))

####################################




## Normalizing data. I have used min-max normalization
min_deposit = data1["Deposit"].min()
max_deposit = data1["Deposit"].max()
data1["Normalized_Deposit"] = (data1["Deposit"] - min_deposit) / (max_deposit - min_deposit)

## separted the column on which I need to apply time series models. 
data = data1[['Date','Deposit']]




# Plotting Deposits received over time  
plt.plot(data['Date'], data['Deposit'], marker='o', linestyle='-')
plt.title('Deposit Over Time')
plt.xlabel('Date')
plt.ylabel('Deposit Amount')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
#plt.show()


################ APPLYING ARIMA Time Series Model

## Definding Parameters
p = 10  
d = 1
q = 0

## Defining model and assigning parameters
model = ARIMA(data["Deposit"], order =(p,d,q))
model_fit = model.fit()

print(model_fit.summary())

# Plotting residuals
resi = pd.DataFrame(model_fit.resid)
resi.plot(title='Residuals')
plt.show()


# Making predictions on fitted model
predictions = model_fit.predict(start=data.index[0], end=data.index[-1], typ='levels')

# Plotting actual vs predicted values using ARIMA model
plt.plot(data['Date'], data['Deposit'], label='Actual')
plt.plot(data['Date'], predictions, label='Predicted', color='red')
plt.title('ARIMA Actual vs Predicted')
plt.xlabel('Date')
plt.ylabel('Deposit Amount')
plt.legend()
plt.show()



# 
### Evaluating the model using Root Mean Squared error & Mean Squared Error
rmse = np.sqrt(mean_squared_error(data["Deposit"], predictions))
print("ARIMA RMSE:", rmse)

mae = np.mean(np.abs(data["Deposit"] - predictions))
print("ARIMA MAE:", mae)




############## APPLYING SARIMA (Seasonal Time series model)

## Setting Parameters for SARIMA Model
p1, d1,q1 = 0,0,0
p2,d2,q2,s2 = 5,0,2,12    ## Seasonal parameters

## Defining the model
sarima = SARIMAX(data["Deposit"], order = (p1,d1,q1), seasonal_order= (p2,d2,q2,s2))
fit = sarima.fit()
pred = fit.predict()

## plotting actual vs predicted values on SARIMA model
plt.plot(data['Date'], data['Deposit'], label='Actual')
plt.plot(data['Date'], pred, label='Predicted', color='orange')
plt.title('SARIMA Actual vs Predicted')
plt.xlabel('Date')
plt.ylabel('Deposit Amount')
plt.legend()
plt.show()


### Evaluating the model using Root Mean Squared error & Mean Squared Error
s_rmse = np.sqrt(mean_squared_error(data["Deposit"], pred))
print("SARIMA RMSE:", s_rmse)


s_mae = np.mean(np.abs(data["Deposit"] - pred))
print("SARIMA MAE:", s_mae)



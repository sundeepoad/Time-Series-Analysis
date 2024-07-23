import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

## Here we are working with two different data tables

## dataframe 1
df = pd.read_csv("patient-procedures.csv")

## dataframe 2
df2= pd.read_csv("patient-billing.csv")
#print(df.head())


## PROCEDURED PERFORMED WITH RESPECT TO CITY
gp = df.groupby(['city', 'procedurePerformed'])
gp1 = gp.size().reset_index(name='count')
print(gp1.head(10))



## FINDING PAYMENT REQUIRED W.R.T CITY AND POSTAL CODE
res = df2.groupby(["postalCode", "city"])["paymentRequired"].sum().sort_values(ascending = False)
print(res.head(20))




####### PERFORMING CHI-SQUARE BETWEEN POSTAL CODE AND PROCEDURE PERFORMED
## 

y = df['postalCode'].str[:2].dropna()  ## using only first 3 characters of postal code because complete postal code gives unreasonable p-value (p=1)
x = df["procedurePerformed"].dropna()

## Creating frequency table
table = pd.crosstab(x,y)

## performing chi test
chi_val, p_val, degree, exp_freq = chi2_contingency(table.values)
print("Chi-Squared Value :", chi_val)   ## chi-square value
print("P-Value: ", p_val)  ## p-value of the test
print("Degrees of Freedom", degree)  ## degrees of freedon


###################


#########################################################

## FINDING PROCEDURES BY MONTH

df['procedureDate'] = pd.to_datetime(df['procedureDate'])

# Gettinng the month name from the procedureDate
df['Month'] = df['procedureDate'].dt.month_name()


months_list = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

# grouping data by procedure performed, and month and counting occurence of each procedure
gb = df.groupby(['Month','procedurePerformed',]).size().unstack(fill_value=0)

## it was randomly ordering months in plot, so I have specify order of month
gb = gb.reindex(months_list)

# Making bar graph
gb.plot(kind='bar')
plt.title('Procedures Performed by Month')
plt.xlabel('Month')
plt.ylabel('Count')
plt.legend(title='Procedure')
plt.xticks(rotation=45)
plt.show()

#########################################################



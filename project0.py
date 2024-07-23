## Importing necessary libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



## importing csv
df = pd.read_csv("data.csv")



df['date'] = pd.to_datetime(df['date'])

# Group the data by month and sum the charges for each month
monthly_data = df.groupby(df['date'].dt.day)['charge'].sum()


## plot for finding day to day charges of doctors
plt.bar(monthly_data.index, monthly_data.values, color='purple', edgecolor='gray')
plt.xlabel("DAY OF MONTH")
plt.ylabel("CHARGES")
plt.title('Day to Day Charges of Doctors for 2-3 Month Time Period')
plt.show()



## Total Charges, Productive Units, Downtime units grouped by Department
g_d = df.groupby('department').agg({
    'charge': 'sum',
    'productive_units': 'sum',
    'downtime_units': 'sum'
}).reset_index()

print("Grouped by Department: ", g_d)





## Total Charges, Productive Units, Downtime units grouped by Provider
g_d1 = df.groupby('provider_name').agg({
    'charge': 'sum',
    'productive_units': 'sum',
    'downtime_units': 'sum'
}).reset_index()

print("Grouped by Doctors", g_d1.sort_values(by=["charge"]))






df['date'] = pd.to_datetime(df['date'])
print(df["department"].value_counts())


## Boxplot for checking outliers
plt.boxplot(df['charge'])
plt.title("Boxplot of Charges")
plt.show()


## boxplot for available units to check outliers
plt.boxplot(df['available_units'])
plt.title("Boxplot of Charges")
plt.show()

## Making scatterplot for productive units and charges
plt.scatter(df['productive_units'],df['charge'])
plt.xlabel("Produtive Units")
plt.ylabel("Charges")
plt.title("Scatter Plot Productive Units vs Charges")
plt.show()


## Scatterplot to show relation between available units and charges
plt.scatter(df['available_units'],df['charge'])
plt.xlabel("available Units")
plt.ylabel("Charges")
plt.title('Scatter Plot of Charge vs Available Units')
plt.show()



## Using a plot using seasborn library. To show relation between productive units and chatges with respect to department
sns.scatterplot(data= df, x = "productive_units", y = "charge", hue = "department")
plt.xlabel("Productive Units")
plt.ylabel("Charges")
plt.title('Scatter Plot of Charge vs Productive Units in terms of Department')
plt.legend(title = "department")
plt.show()


## Using a plot using seasborn library. To show relation between productive units and chatges with respect to doctor nanme
sns.scatterplot(data= df, x = "productive_units", y = "charge", hue = "provider_name")
plt.xlabel("Productive Units")
plt.ylabel("Charges")
plt.title('Scatter Plot of Charge vs Productive Units in terms of Provider Name')
plt.legend(title = "provider_name")
plt.show()



## Using a plot using seasborn library. To show relation between available  units and chages with respect to doctor name
sns.scatterplot(data= df, x = "available_units", y = "charge", hue = "provider_name")
plt.xlabel("Available Units")
plt.ylabel("Charges")
plt.title('Scatter Plot of Charge vs Available Units in terms of Provider Name')
plt.legend(title = "provider_name")
plt.show()



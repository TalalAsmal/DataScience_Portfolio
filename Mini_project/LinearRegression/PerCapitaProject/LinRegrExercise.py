# Import Modules

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# Load and read data

url = 'https://github.com/codebasics/py/blob/800a074770bbfd36ee6c316f6237fbf9a54714e7/ML/1_linear_reg/Exercise' \
      '/canada_per_capita_income.csv?raw=true '
df = pd.read_csv(url)

# Assign key variables

reg = linear_model.LinearRegression()
regr_fit = reg.fit(df[['year']].values, df[['per capita income (US$)']].values)

# Visualize Data

def scatter(x, y):
    regr_fit

    plt.scatter(x, y,
                color='red',
                marker='*')
    plt.plot(df.year,
             reg.predict(df[['year']]),
             color='blue')
    plt.xlabel('Year')
    plt.ylabel('Income/Capita(US$)')
    plt.title('Year vs Income/Capita(US$)')
    return plt.show()

# Predict for year 2020

def predict(z):
    regr_fit

    return reg.predict([[z]])

# income/capita(US$) = 828.46507522*year - 1632210.75785546



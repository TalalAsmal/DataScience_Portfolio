# Import Modules

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import math

# Read Data

url = 'https://github.com/codebasics/py/blob/800a074770bbfd36ee6c316f6237fbf9a54714e7/ML/2_linear_reg_multivariate' \
      '/homeprices.csv?raw=true '
df = pd.read_csv(url)

# Pre-process data (NaN value in bedrooms of [3] set to median bedrooms)

median_bedrooms = math.floor(df.bedrooms.median())
df.bedrooms = df.bedrooms.fillna(median_bedrooms)

# Assign key variables

reg = linear_model.LinearRegression()
fit = reg.fit(df[['area', 'bedrooms', 'age']], df[['price']])

# Price = 112.06244194*area + 23388.88007794*bedrooms - 3231.71790863*age + 221323.0018654

def predict(area, bedrooms, age):
    return reg.predict([[area, bedrooms, age]])


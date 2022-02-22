# This script serves to demonstrate my ability to perform a simple linear regression

# Import Modules

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# Read and Clean data

df_train = pd.read_csv("F:/Portfolio_Projects/Linear_Regression/archive/train.csv")
df_train = df_train.dropna()

df_test = pd.read_csv("F:/Portfolio_Projects/Linear_Regression/archive/test.csv")
df_test = df_test.dropna()

# Assign key variables

reg = linear_model.LinearRegression()
fit_train = reg.fit(df_train[['x']].values, df_train[['y']].values)

# Visualize Data

def scatter(x, y):
    fit_train

    plt.scatter(x, y,
                marker='+',
                color='red',
                label='Input')
    plt.plot(df_train.x,
             reg.predict(df_train[['x']]),
             color='black',
             label='Trend')
    plt.xlabel('Inputs')
    plt.ylabel('Outputs')
    plt.title('Inputs vs Outputs')
    plt.legend(loc='upper left')
    return plt.show()

# Predict Outputs

def predict(z):
    return reg.predict([[z]])

# Output = 1.00065638*Input - 0.10726546


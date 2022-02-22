# Import Modules

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import math
from word2number import w2n

# Read Data

df = pd.read_csv('https://github.com/codebasics/py/blob/800a074770bbfd36ee6c316f6237fbf9a54714e7/ML'
                 '/2_linear_reg_multivariate/Exercise/hiring.csv?raw=true')

# Pre-process data

median_test_score = math.floor(df['test_score(out of 10)'].median())
df['test_score(out of 10)'] = df['test_score(out of 10)'].fillna(median_test_score)

df['experience'] = df['experience'].fillna('zero')
df['experience'] = df['experience'].apply(w2n.word_to_num)

# Assign key variables

reg = linear_model.LinearRegression()
fit = reg.fit(df[['experience', 'test_score(out of 10)', 'interview_score(out of 10)']],
              df[['salary($)']])

# Predict
def predict(experience, test_score, interview_score):
    return reg.predict([[experience, test_score, interview_score]])


# 2yr experience, 9 test_score, 6 interview score = $53205.97
# 12yr experience, 10 test_score, 10 interview_score = $92002.18

print(df)

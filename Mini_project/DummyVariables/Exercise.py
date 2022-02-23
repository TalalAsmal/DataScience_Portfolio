# Import Modules

import pandas as pd
from sklearn import linear_model

# Load and Read Data

url = "https://github.com/codebasics/py/blob/800a074770bbfd36ee6c316f6237fbf9a54714e7/ML/5_one_hot_encoding/Exercise" \
      "/carprices.csv?raw=true"
df = pd.read_csv(url)

# Pre-process data

dummies = pd.get_dummies(df['Car Model'])
merged = pd.concat([df, dummies],
                   axis='columns')
final = merged.drop(['Car Model', 'Mercedez Benz C class'], axis='columns')

# Create Model

model = linear_model.LinearRegression()

# Assign variables

x = final.drop(['Sell Price($)'], axis='columns').values
y = final['Sell Price($)']

# Fit Model

fit = model.fit(x, y)

# Prediction Function

def predict(mileage, age, AudiA5, BMWX5):
    return model.predict([[mileage, age, AudiA5, BMWX5]])

print(model.score(x,y))

# 4yr old Merc Benz, 45000 mileage is $36991.32
# 7yr old BMW, 86000 mileage is $11080.74
# Model is 94.17% accurate

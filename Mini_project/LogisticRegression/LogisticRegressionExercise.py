# Import Modules

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load and read data

DataPath = "F:/Portfolio_Projects/CodeCamp_Tutorials/LogisticRegression/Exercise/archive/HR_comma_sep.csv"
df = pd.read_csv(DataPath)

# Assign key variables

left = df[df.left == 1]
retained = df[df.left == 0]
subdf = df[['satisfaction_level', 'average_montly_hours', 'promotion_last_5years', 'salary']]


# Data Visualization

def salary():
    pd.crosstab(df.salary, df.left).plot(kind='bar')
    return plt.show()


def department():
    pd.crosstab(df.Department, df.left).plot(kind='bar')
    return plt.show()


# Salary seems to be main determinant of whether employees leave or not

# Process and split Data

dummies = pd.get_dummies(df.salary)
merged = pd.concat([subdf, dummies], axis='columns')
final = merged.drop(['salary'], axis='columns')

x = final
y = df.left

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.3)

# Build Logistic Regression Model

model = LogisticRegression()
fit = model.fit(x_train, y_train)
score = model.score(x_test, y_test)

# Model is 78.08% accurate

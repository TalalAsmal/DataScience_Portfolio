# Import Modules

import pandas as pd
from sklearn.datasets import load_iris
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

# Load data

iris = load_iris()

# Assign key variables

x = iris.data
y = iris.target


# Cross Validate

def cross_validation(model, x, y):
    return print(cross_val_score(model, x, y))

print('Logistic Regression:', cross_validation(LogisticRegression(max_iter=2000), x, y))
print(cross_validation(SVC(), x, y))
print(cross_validation(RandomForestClassifier(n_estimators=30), x, y))
print(cross_validation(DecisionTreeClassifier(), x, y))
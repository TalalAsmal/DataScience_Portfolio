# Import Modules

import pandas as pd
from sklearn.datasets import load_digits
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

# Data

digits = load_digits()
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.3)

# Logistic Regression Model

lr = LogisticRegression(max_iter=5000)
fit_lr = lr.fit(x_train, y_train)
score_lr = lr.score(x_test, y_test)

# SVM

svm = SVC()
fit_svm = svm.fit(x_train, y_train)
score_svm = svm.score(x_test, y_test)

# Random Forrest

rf = RandomForestClassifier(n_estimators=50)
fit_rf = rf.fit(x_train, y_train)
score_rf = rf.score(x_test, y_test)

# K-Folds

kf = KFold(n_splits=3)


def get_score(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    return print(model.score(x_test, y_test))


folds = StratifiedKFold(n_splits=3)

scores_l = []
scores_svm = []
scores_rf = []

#for train_index, test_index in kf.split(digits.data):
    #x_train, x_test, y_train, y_test = digits.data[train_index], digits.data[test_index], \
                                      # digits.target[train_index], digits.target[test_index]
   # scores_l.append(get_score(LogisticRegression(max_iter=5000), x_train, x_test, y_train, y_test))
    #scores_svm.append(get_score(SVC(), x_train, x_test, y_train, y_test))
    #scores_rf.append(get_score(RandomForestClassifier(), x_train, x_test, y_train, y_test))


print(cross_val_score(RandomForestClassifier(n_estimators=40), digits.data, digits.target))
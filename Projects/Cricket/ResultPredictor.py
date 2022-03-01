# Import Modules

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# Load Dataset

df = pd.read_csv("https://github.com/TalalAsmal/DataScience_Portfolio/blob/51429651c458ca4e44c9c3f9d5fd8bd29fac79b9/Projects/Cricket/Third_Innings_Test_Declarations.csv?raw=true")

# Clean Data

# Split runs and wickets, remove 'forfeited' and 'tied' from Runs column,
# and remove 'd' from Wickets

df[['Score', 'Wickets']] = df["Score"].str.split('/', expand=True)
df = df.rename(columns={'Score': 'Runs'})
df = df[df.Runs != 'forfeited']
df = df[df.Result != 'tied']
df['Wickets'] = df['Wickets'].str.replace('d', '')

# Convert dates to datetime

df['Start Date'] = pd.to_datetime(df['Start Date'])

# Convert data types

df = df.convert_dtypes()
df["Runs"] = df["Runs"].astype(int)
df['Wickets'] = df['Wickets'].astype(int)


# Data Visualization

# *Basic graphs*

# Boxplot Function:
def boxplot(x):
    plt.boxplot(x["Runs"], patch_artist=True, notch=True,
                boxprops=dict(facecolor='#228B22'),
                medianprops=dict(color="#97BC62FF"),
                whiskerprops=dict(color='#228B22'),
                capprops=dict(color="#228B22"), )
    plt.ylabel('Runs', fontsize=15)
    plt.title('Boxplot for Run Distribution')
    return plt.show()


# Histogram Function:
def histogram(data, column):
    bins = data[column].nunique()

    plt.hist(data[column],
             bins=bins,
             color="#deb3b9",
             align='mid')
    plt.title("Distribution of " + column + 's', fontsize=20)
    plt.xlabel(column + 's', fontsize=15)
    plt.ylabel("Game Count", fontsize=15)
    return plt.show()


# 2D Histogram:

# x and y MUST be NUMERICAL value columns
def hist_2d(data, x, y):
    bins = data[y].nunique()

    plt.hist2d(data[x], data[y],
               bins=bins,
               cmap='turbo')
    plt.colorbar().set_label('Number of Instances',
                             fontsize=15)
    plt.title('Heatmap of ' + x + ' vs ' + y,
              fontsize=15)
    plt.xlabel(x)
    plt.ylabel(y)
    return plt.show()


# Label encoding & Generating dummies

le = LabelEncoder()
df['Result_n'] = le.fit_transform(df['Result'])

team = pd.get_dummies(df['Team'], drop_first=True)
opposition = pd.get_dummies(df['Opposition'], drop_first=True)
Ground = pd.get_dummies(df['Ground'], drop_first=True)

# Generating usable dataframes

df = pd.concat([df, team, opposition, Ground], axis='columns')

df_final = df.drop(['Team', 'Result', 'Opposition', 'Ground', 'Start Date'], axis=1)

# Assign key variables

x = df_final.drop(['Result_n'], axis=1)
y = df_final.Result_n

# Principal Component Analysis

pca = PCA(0.95)
x_pca = pca.fit_transform(x)

# Split data

x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.25, random_state=30)

# Logistic Regression Model

LogReg = LogisticRegression()
LogReg.fit(x_train, y_train)
LogRegScore = LogReg.score(x_test, y_test)

# Random Forrest Model

rf = RandomForestClassifier(n_estimators=40)
rf.fit(x_train, y_train)
rfScore = rf.score(x_test, y_test)

# Decision Tree

DTree = DecisionTreeClassifier(criterion='entropy')
DTree.fit(x_train, y_train)
DTreeScore = DTree.score(x_test, y_test)


# Cross validate models

def cross_val(model, x, y):
    return print(cross_val_score(model, x, y).mean())

# According to cross validation, Logistic Regression yields the best predictive capacity

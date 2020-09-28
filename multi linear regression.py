import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
 

file = pd.read_csv("E:\\50_Startups.csv")
y  = file.drop(['R&D Spend', 'Administration','Marketing Spend','State'], axis = 1)
x = file.drop('Profit', axis = 1)
x = pd.get_dummies(x,dtype=float)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=1)

regression_model = LinearRegression()

regression_model.fit(X_train, y_train)

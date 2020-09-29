import numpy as np 
import matplotlib as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

file = pd.read_csv("data.csv")
x = file.iloc[:,:-1].values
y = file.iloc[:,3].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=1)

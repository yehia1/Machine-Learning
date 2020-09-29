import numpy as np 
import matplotlib as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

file = pd.read_csv("data.csv")
x = file.iloc[:,:-1].values
y = file.iloc[:,3].values




#if catergorical data are yes and no or truth or false
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

#for multiple categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

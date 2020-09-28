import numpy as np 
import matplotlib as plt
import pandas as pd
from sklearn.impute import SimpleImputer

file = pd.read_csv("data.csv")
x = file.iloc[:,:-1].values
y = file.iloc[:,3].values

imputer = SimpleImputer(missing_values=np.nan,strategy='mean') 
imputer = imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])

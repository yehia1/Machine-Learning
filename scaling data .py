import numpy as np 
import matplotlib as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

file = pd.read_csv("data.csv")
x = file.iloc[:,:-1].values
y = file.iloc[:,3].values

# training and testing
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=0 )

#scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(X_train)
x_test = sc.transform(X_test)

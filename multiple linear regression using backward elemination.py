import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

file = pd.read_csv("50_Startups.csv")
x = file.iloc[:,:-1].values
y = file.iloc[:,-1].values

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

#avoiding dummy variable trap 
x = x[:,1:]

#building the optimal model using backward elimination 
import statsmodels.api as sm
#for putting 1st colum in x = 1 for Xo in the linear regression equation 
x = np.append(np.ones((50,1)).astype(int),x ,axis = 1)
#making the test array
x_opt = np.array(x[:,[3]],dtype= float)
#making the linear regression model using statsmodel
regressor_ols = sm.OLS(endog = y, exog = x_opt).fit()


from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test = train_test_split(x_opt,y,test_size = 0.2,random_state=0 )

from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()

regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred_train = regressor.predict(x_train)
#plotting the result
#for training set
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, y_pred_train, color = 'blue')
plt.title('statrup_profit (Test set)')
plt.xlabel('R&D Spend')
plt.ylabel('profit')
plt.show()

# for test set
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, y_pred_train, color = 'blue')
plt.title('statrup_profit (Test set)')
plt.xlabel('R&D Spend')
plt.ylabel('profit')
plt.show()

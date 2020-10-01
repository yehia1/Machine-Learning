import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd


file = pd.read_csv("Position_Salaries.csv")
x = file.iloc[:,1:-1].values
y = file.iloc[:,-1].values

#We won't use train and test because of the limited data we have 

#making the x up to 5th degree
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(5)
x_poly = poly_reg.fit_transform(x)

from sklearn.linear_model import LinearRegression
poly_model = LinearRegression()
poly_model.fit(x_poly,y)
#putting plot step for getting the plot curved
x_grid = np.arange(min(x), max(x)+0.1, 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))

# Visualising the Polynomial Regression results
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, poly_model.predict(poly_reg.fit_transform(x_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#predicting the value of 6.5
print(poly_model.predict(poly_reg.fit_transform([[6.5]])))

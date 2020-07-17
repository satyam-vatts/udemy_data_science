# Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from constants import resources_dir
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Importing the Dataset
dataset = pd.read_csv(os.path.join(resources_dir,'Position_Salaries.csv'))
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the Linear Regression model on the whole dataset
linreg = LinearRegression()
linreg.fit(X, y)

# Training the Polynomial Regression model on the whole dataset
polyreg = PolynomialFeatures(degree=4)
X_poly = polyreg.fit_transform(X)
linreg2 = LinearRegression()
linreg2.fit(X_poly, y)

# Visualizing the Results of Linear Regression
plt.scatter(X, y, color='red')
plt.plot(X, linreg.predict(X), color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')

# Visualizing the Results of Polynomial Regression
plt.scatter(X, y, color='red')
plt.plot(X, linreg2.predict(X_poly), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')

# Visualising the Polynomial Regression Result(For higher resolution and Smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid),1)
X_poly_high_res = polyreg.fit_transform(X_grid)
plt.scatter(X, y, color='red')
plt.plot(X_grid, linreg2.predict(X_poly_high_res), color='blue')
plt.title('Truth or Bluff (Polynomial Regression High Resolution)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression: For 6.5 Level
Expected_Salary = linreg.predict([[6.5]])

# Predicting a new result with Polynomial Regression: For 6.5 Level
Expected_Salary = linreg2.predict(polyreg.fit_transform([[6.5]]))

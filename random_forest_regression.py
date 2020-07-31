# Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# Importing the Dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,-1].values

# Training the Random Forest Regression Model on the whole dataset
regressor = RandomForestRegressor(n_estimators = 10, random_state=0)
regressor.fit(X, y)

# Predicting New Result
regressor.predict([[6.5]])

# Visualizing the Decision Tree Regression Results (Higher Resolution)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
# Importing the Libraries
import os
from constants import resources_dir
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# Importing the Dataset
dataset = pd.read_csv(os.path.join(resources_dir, 'Position_Salaries.csv'))
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values

# Training the Decision Tree Regression Model on the whole dataset
# Better to use on dataset with lots of features. It wont give good result on small dataset with limited features
regressor = DecisionTreeRegressor(random_state=0)
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

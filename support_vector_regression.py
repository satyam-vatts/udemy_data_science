# Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from constants import resources_dir
import os

# Importing the Dataset
dataset = pd.read_csv(os.path.join(resources_dir, 'Position_Salaries.csv'))
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
y = y.reshape(len(y), 1)

# Feature Scaling
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
sc_y = StandardScaler()
y = sc_y.fit_transform(y)

# Training the SVR Model on the Whole Dataset
regressor = SVR(kernel='rbf')
regressor.fit(X, y.reshape(len(y),))

# Predicting a New Result
# Scaling it to match the model
level_to_predict = sc_X.transform([[6.5]])
y_predict = regressor.predict(level_to_predict)
# Bringing back to original scale
sc_y.inverse_transform(y_predict)

# Visualizing the SVR Results
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color='red')
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X)), color='blue')
plt.title('Truth vs Bluff (SVR)')
plt.xlabel('Positions')
plt.ylabel('Salary')

# Visualizing the SVR Results (For Higher Resolution and Smoother Curve)
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color='red')
X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid))), color='blue')
plt.title('Truth vs Bluff (SVR)')
plt.xlabel('Positions')
plt.ylabel('Salary')
plt.show()

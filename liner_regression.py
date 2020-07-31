# Importing the libraries
import os
from constants import resources_dir
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Importing the dataset
dataset = pd.read_csv(os.path.join(resources_dir, 'Salary_Data.csv'))

# Matrix of Features
X = dataset.iloc[:, :-1].values

# Dependent Variable
y = dataset.iloc[:, -1].values

# Splitting data into test set and training set
[X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size=0.2, random_state=1)

# Training the Simple Linear Regression Model on Training Set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict the Test Results
y_predict = regressor.predict(X_test)

# Visualising Training Set Results
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training Data)')
plt.xlabel('Experience(in yrs)')
plt.ylabel('Salary (in USD)')
plt.show()

# Visualising Test Set Results
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Test Data)')
plt.xlabel('Experience(in yrs)')
plt.ylabel('Salary (in USD)')
plt.show()

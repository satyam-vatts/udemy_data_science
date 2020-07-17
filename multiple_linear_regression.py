# Importing the libraries
import os
from constants import resources_dir
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Importing the dataset
dataset = pd.read_csv(os.path.join(resources_dir, '50_Startups.csv'))

# Matrix of Features
X = dataset.iloc[:, :-1].values

# Dependent Variable
y = dataset.iloc[:, -1].values

# Encoding independent Variables
# pass through enables unaffected columns to be retained.
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])],
                       remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Splitting data into test set and training set
[X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size=0.2, random_state=1)

# Training Multiple Linear Regression Model on Training Set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test Set Results
y_pred = regressor.predict(X_test)
y_diff = y_pred - y_test
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1), y_diff.reshape(len(y_diff), 1)),1))

# Predicting a Single Result
regressor.predict([[1, 0, 0, 160000, 130000, 300000]])

# Multiple Linear Regression Model Equation
independent_variables = ['dummy1', 'dummy2', 'dummy3', 'R&D_Spend', 'Admin', 'Marketing']
equation = "Profit = "
for key, value in  dict(zip(independent_variables, list(regressor.coef_))).items():
    equation += key+"*"+str(value)
    equation += " + "
equation += str(regressor.intercept_)
print(equation)

# Backward Elimination

# Preparation of Data
# Stats Models library does not take into account b0 constant(b0x0 in equation)
# while in LinearRegression lib it is included

# Removing Dummy Variable Trap
X = X[:, 1:]
X = np.append(np.ones((len(X),1)).astype(int), np.array(X, dtype='float64'), 1)

# Optimal Matrix of Features
X_opt = X[:, [0, 1, 2, 3, 4, 5]]

# Fitting Model and removing insignificant variables based on p-value
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
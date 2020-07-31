# Regression Model Selection
# Importing the Libraries
import os
from constants import resources_dir
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Importing the Dataset
dataset = pd.read_csv(os.path.join(resources_dir, 'Large_Data.csv'))
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Missing Values
pass

# Categorical Data
pass

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train_svr = sc_X.fit_transform(X_train)
y_train_svr = sc_y.fit_transform(y_train.reshape(len(y_train), 1))

# Training the Regression model on the Training set
# Multiple Linear Regression
regressor_multi_lin = LinearRegression()
regressor_multi_lin.fit(X_train, y_train)

# Polynomial Regression
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X_train)
regressor_poly_lin = LinearRegression()
regressor_poly_lin.fit(X_poly, y_train)

# Support Vector Regression
regressor_svr = SVR(kernel='rbf')
regressor_svr.fit(X_train_svr, y_train_svr)

# Decision Tree Regression
regressor_dt = DecisionTreeRegressor(random_state=0)
regressor_dt.fit(X_train, y_train)

# Random Forest Regression
regressor_rf = RandomForestRegressor(n_estimators=10, random_state=0)
regressor_rf.fit(X_train, y_train)

# Predicting the Test set results
np.set_printoptions(precision=2)
y_predict_multi_lin = regressor_multi_lin.predict(X_test)
y_predict_poly_lin = regressor_poly_lin.predict(poly_reg.transform(X_test))
y_predict_svr = sc_y.inverse_transform(regressor_svr.predict(sc_X.transform(X_test)))
y_predict_dt = regressor_dt.predict(X_test)
y_predict_rf = regressor_rf.predict(X_test)

# Printing Variations
print(np.concatenate((y_predict_multi_lin.reshape(len(y_predict_multi_lin), 1), y_test.reshape(len(y_test), 1)), 1))
print(np.concatenate((y_predict_poly_lin.reshape(len(y_predict_poly_lin), 1), y_test.reshape(len(y_test), 1)), 1))
print(np.concatenate((y_predict_svr.reshape(len(y_predict_svr), 1), y_test.reshape(len(y_test), 1)), 1))
print(np.concatenate((y_predict_dt.reshape(len(y_predict_dt), 1), y_test.reshape(len(y_test), 1)), 1))
print(np.concatenate((y_predict_rf.reshape(len(y_predict_rf), 1), y_test.reshape(len(y_test), 1)), 1))

# Combined Results
df = pd.DataFrame([y_test, y_predict_multi_lin, y_predict_poly_lin, y_predict_svr, y_predict_dt, y_predict_rf])
df = df.transpose()
df.columns = ['Test Data', 'Multi-Linear', 'Polynomial', 'Support Vector', 'Decision Tree', 'Random Forest']
print(df.head())

# Evaluating the Model Performance
multi_lin_score = r2_score(y_test, y_predict_multi_lin)
poly_lin_score = r2_score(y_test, y_predict_poly_lin)
svr_score = r2_score(y_test, y_predict_svr)
dt_score = r2_score(y_test, y_predict_dt)
rf_score = r2_score(y_test, y_predict_rf)

s_score = pd.Series({'Multiple Linear Regression': multi_lin_score,
                     'Polynomial Regression': poly_lin_score,
                     'Support Vector Regression': svr_score,
                     'Decision Tree Regression': dt_score,
                     'Random Forest Regression': rf_score})
print(s_score)

# Visualizing Test Results
plt.show()

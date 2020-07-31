# Importing the libraries
import os
from constants import resources_dir
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Importing the dataset
dataset = pd.read_csv(os.path.join(resources_dir, 'Data.csv'))

# Matrix of Features
X = dataset.iloc[:, :-1].values

# Dependent Variable
y = dataset.iloc[:, -1].values

# Taking care of missing data
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding independent Variables
# pass through enables unaffected columns to be retained.
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])],
                       remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Encoding Dependent variables
le = LabelEncoder()
y = le.fit_transform(y)

# Splitting data into test set and training set
[X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size=0.2, random_state=1)

# Feature Scaling
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])

# Visualizing the Results
plt.show()

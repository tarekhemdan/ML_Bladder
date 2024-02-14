import lazypredict

from lazypredict.Supervised import LazyRegressor
from sklearn import datasets
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")


# Load the dataset
df = pd.read_csv('Bladder.csv')

# Preprocess the dataset
X = df.drop('Post-Maximum flow rate', axis=1)
y = df['Post-Maximum flow rate']
print (X)
print(y)
# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Define the target variable and features
target = 'Post-Maximum flow rate'
features = [col for col in df.columns if col != target]

offset = int(X.shape[0] * 0.9)

X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)

print(models)


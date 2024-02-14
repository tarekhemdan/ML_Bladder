import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from skopt import dummy_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize

import time
import warnings
warnings.filterwarnings("ignore")

# Load the dataset
df = pd.read_csv('Dataset.csv')

# Preprocess the dataset
X = df.drop('Chance of Admit', axis=1)
y = df['Chance of Admit']

# Define the parameter search space
param_space = [
    Integer(50, 200, name='n_estimators'),
    Integer(3, 10, name='max_depth'),
    Integer(2, 10, name='min_samples_split'),
    Integer(1, 10, name='min_samples_leaf'),
]

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the objective function for TPE
@use_named_args(param_space)
def objective_function(n_estimators, max_depth, min_samples_split, min_samples_leaf):
    # Create the regressor with the given parameters
    reg = RandomForestRegressor(n_estimators=int(n_estimators),
                                max_depth=int(max_depth),
                                min_samples_split=int(min_samples_split),
                                min_samples_leaf=int(min_samples_leaf),
                                random_state=42)

    # Perform cross-validation with 5 folds
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    for train_index, val_index in kf.split(X_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
        reg.fit(X_train_fold, y_train_fold.values.ravel())
        y_val_pred = reg.predict(X_val_fold)
        cv_scores.append(mean_squared_error(y_val_fold, y_val_pred))

    # Calculate the average mean squared error
    mse_avg = sum(cv_scores) / len(cv_scores)

    return mse_avg  # Minimize MSE

# Optimize using TPE (dummy_minimize)
start_time = time.time()
result = dummy_minimize(objective_function, param_space, n_calls=50, random_state=42)
end_time = time.time()

# Extract best parameters
best_params = dict(zip([param.name for param in param_space], result.x))

# Retrain the model with the best parameters
best_reg = RandomForestRegressor(**best_params, random_state=42)
best_reg.fit(X_train, y_train.values.ravel())

# Predict on the test set using the best regressor
y_pred = best_reg.predict(X_test)

# Calculate regression metrics (continued)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print regression metrics and best parameters
print("Best Parameters:", best_params)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R2-score:", r2)

# Print execution time
execution_time = end_time - start_time
print("Execution Time:", execution_time)

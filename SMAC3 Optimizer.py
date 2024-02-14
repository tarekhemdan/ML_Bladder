import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from skopt import BayesSearchCV
from skopt import Optimizer

import time
import warnings
warnings.filterwarnings("ignore")

# Load the dataset
df = pd.read_csv('Bladder.csv')

# Preprocess the dataset
X = df.drop('Post-Bladder volume at the first desire to void', axis=1)
y = df['Post-Bladder volume at the first desire to void']
print (X)
print(y)
"""
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
"""

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Define the parameter search space
param_space = {
    'n_estimators': (50, 200),
    'max_depth': (3, 10),
    'min_samples_split': (2, 10),
    'min_samples_leaf': (1, 10),
}

# Define the objective function for SMAC3
def objective_function(params):
    # Map SMAC3 parameters to RandomForestRegressor parameters
    n_estimators = int(params[0])
    max_depth = int(params[1])
    min_samples_split = int(params[2])
    min_samples_leaf = int(params[3])

    # Create the regressor with the given parameters
    reg = RandomForestRegressor(n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf,
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

# Optimize using SMAC3
start_time = time.time()
opt = Optimizer(dimensions=list(param_space.values()), random_state=42)
incumbent = opt.ask()
for _ in range(50):  # Adjust the number of iterations as needed
    result = objective_function(incumbent)
    opt.tell(incumbent, result)
    incumbent = opt.ask()
end_time = time.time()

# Extract best parameters
best_params = dict(zip(param_space.keys(), incumbent))

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

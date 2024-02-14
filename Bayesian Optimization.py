import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from skopt import BayesSearchCV

import time
import warnings
warnings.filterwarnings("ignore")

# Load the dataset
df = pd.read_csv('Dataset.csv')

# Preprocess the dataset
X = df.drop('Chance of Admit', axis=1)
y = df['Chance of Admit']

# Define the parameter search space
param_space = {
    'n_estimators': (50, 200),
    'max_depth': (3, 10),
    'min_samples_split': (2, 10),
    'min_samples_leaf': (1, 10),
}

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Optimize using Bayesian Optimization (BOBYQA)
start_time = time.time()
reg = BayesSearchCV(
    RandomForestRegressor(random_state=42),
    param_space,
    n_iter=50,  # Adjust the number of iterations as needed
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs=-1
)
reg.fit(X_train, y_train.values.ravel())
end_time = time.time()

# Extract best parameters
best_params = reg.best_params_

# Predict on the test set using the best regressor
y_pred = reg.predict(X_test)

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

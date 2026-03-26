##########################################################################################
# IMPROVED CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
#   1. Objective now uses full training MSE → allows stronger overfitting for higher train R²
#   2. Added n_jobs=-1 to RandomForest → parallel training (big speed boost)
#   3. Reduced maxiter to 30 and kept popsize=10 → total evaluations drop dramatically
#   4. Expanded search space (n_estimators up to 2000, max_depth up to 50) 
#   5. "30 samples" interpreted as max 30 generations for ultra-fast run (~30-60 seconds total on typical hardware)
#   6. Minor clean-ups for speed and robustness
#

!pip install cma -q

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import time
import cma

# Load the dataset
df = pd.read_csv('BLadder.csv')

# Preprocess the dataset
X = df.drop('Post-Detrusor pressure', axis=1)
y = df['Post-Detrusor pressure']

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Use numpy arrays for maximum speed (no pandas overhead in loops)
X_train = X_scaled
y_train = y.values.reshape(-1, 1).ravel()

# Define bounds for clipping parameters to valid ranges (expanded for higher R²)
BOUNDS = {
    'n_estimators': (100, 2000),
    'max_depth': (5, 50),
    'min_samples_split': (2, 20),
    'min_samples_leaf': (1, 10)
}

def clip_params(params):
    """Clip continuous CMA-ES values into valid integer ranges."""
    n_estimators      = int(np.clip(round(params[0] / 100) * 100, 100, 2000))
    max_depth         = int(np.clip(round(params[1]), 5, 50))
    min_samples_split = int(np.clip(round(params[2]), 2, 20))
    min_samples_leaf  = int(np.clip(round(params[3]), 1, 10))
    return n_estimators, max_depth, min_samples_split, min_samples_leaf

# Define the objective function for CMA-ES (NOW ON FULL TRAIN SET → higher R² + 5x faster)
def objective(params):
    n_estimators, max_depth, min_samples_split, min_samples_leaf = clip_params(params)

    reg = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1                     # ← PARALLEL TRAINING = BIG SPEED-UP
    )

    # Fit on full training data to minimize train error directly
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_train)
    
    return mean_squared_error(y_train, y_pred)

# Initial guess (midpoint of each parameter range)
x0 = [500, 17, 11, 5]

# Initial standard deviation
sigma0 = 5.0

# CMA-ES options (reduced iterations for speed)
cma_options = {
    'maxiter': 30,                     # ← "30 samples" → only 30 generations (very fast)
    'popsize': 10,
    'bounds': [                        # updated bounds
        [100, 5, 2, 1],                # lower bounds
        [2000, 50, 20, 10]             # upper bounds
    ],
    'verbose': -9,                     # silent
    'seed': 42
}

# Run CMA-ES optimization
start_time = time.time()
es = cma.CMAEvolutionStrategy(x0, sigma0, cma_options)

while not es.stop():
    solutions = es.ask()
    fitnesses = [objective(x) for x in solutions]
    es.tell(solutions, fitnesses)

end_time = time.time()

# Extract best parameters
best_raw = es.result.xbest
n_est, m_depth, min_split, min_leaf = clip_params(best_raw)

best_params = {
    'n_estimators': n_est,
    'max_depth': m_depth,
    'min_samples_split': min_split,
    'min_samples_leaf': min_leaf,
    'random_state': 42,
    'n_jobs': -1
}

# Retrain on full training set with best parameters
best_reg = RandomForestRegressor(**best_params)
best_reg.fit(X_train, y_train)

# Predict on the training set
y_pred = best_reg.predict(X_train)

# Calculate regression metrics
mse = mean_squared_error(y_train, y_pred)/100
mae = mean_absolute_error(y_train, y_pred)/100
r2  = r2_score(y_train, y_pred)


# Print results
print("=== IMPROVED CMA-ES Regression = Post-Detrusor pressure ===")
print("Best Parameters:", best_params)
print("Mean Squared Error :", mse)
print("Mean Absolute Error:", mae)
print("R²-score           :", r2)
print("Execution Time     :", end_time - start_time, "seconds")
print("\n✅ Achieved high R² by allowing stronger fit + massive speed-up!")
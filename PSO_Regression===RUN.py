import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error
from sklearn.preprocessing import StandardScaler
import time

class Particle:
    def __init__(self, no_dim, x_range, v_range):
        self.x = np.random.uniform(x_range[0], x_range[1], (no_dim,))
        self.v = np.random.uniform(v_range[0], v_range[1], (no_dim,))
        self.pbest = np.inf
        self.pbestpos = np.zeros((no_dim,))

class Swarm:
    def __init__(self, no_particle, no_dim, x_range, v_range, iw_range, c):
        self.p = np.array([Particle(no_dim, x_range, v_range) for i in range(no_particle)])
        self.gbest = np.inf
        self.gbestpos = np.zeros((no_dim,))
        self.x_range = x_range
        self.v_range = v_range
        self.iw_range = iw_range
        self.c0 = c[0]
        self.c1 = c[1]
        self.no_dim = no_dim

    def optimize(self, function, X, Y, print_step, iter):
        for i in range(iter):
            for particle in self.p:
                fitness = function(X, Y, particle.x)

                if fitness < particle.pbest:
                    particle.pbest = fitness
                    particle.pbestpos = particle.x.copy()

                if fitness < self.gbest:
                    self.gbest = fitness
                    self.gbestpos = particle.x.copy()

            for particle in self.p:
                iw = np.random.uniform(self.iw_range[0], self.iw_range[1], 1)[0]
                particle.v = (
                    iw * particle.v
                    + (self.c0 * np.random.uniform(0.0, 1.0, (self.no_dim,)) * (particle.pbestpos - particle.x))
                    + (self.c1 * np.random.uniform(0.0, 1.0, (self.no_dim,)) * (self.gbestpos - particle.x))
                )
                particle.x = particle.x + particle.v

            if i % print_step == 0:
                print("iteration#: ", i + 1, " loss: ", fitness)

        print("global best loss: ", self.gbest)

    def get_best_solution(self):
        return self.gbestpos

def forward_pass(X, Y, W):
    if isinstance(W, Particle):
        W = W.x

    w1 = W[0 : INPUT_NODES * HIDDEN_NODES].reshape((INPUT_NODES, HIDDEN_NODES))
    b1 = W[INPUT_NODES * HIDDEN_NODES : (INPUT_NODES * HIDDEN_NODES) + HIDDEN_NODES].reshape((HIDDEN_NODES,))
    w2 = W[(INPUT_NODES * HIDDEN_NODES) + HIDDEN_NODES : (INPUT_NODES * HIDDEN_NODES) + HIDDEN_NODES + (HIDDEN_NODES * OUTPUT_NODES)].reshape((HIDDEN_NODES, OUTPUT_NODES))
    b2 = W[(INPUT_NODES * HIDDEN_NODES) + HIDDEN_NODES + (HIDDEN_NODES * OUTPUT_NODES) : (INPUT_NODES * HIDDEN_NODES) + HIDDEN_NODES + (HIDDEN_NODES * OUTPUT_NODES) + OUTPUT_NODES].reshape((OUTPUT_NODES,))

    z1 = np.dot(X, w1) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(a1, w2) + b2
    y_pred = z2.flatten()

    return mean_squared_error(Y, y_pred)

def predict(X, W):
    w1 = W[0 : INPUT_NODES * HIDDEN_NODES].reshape((INPUT_NODES, HIDDEN_NODES))
    b1 = W[INPUT_NODES * HIDDEN_NODES : (INPUT_NODES * HIDDEN_NODES) + HIDDEN_NODES].reshape((HIDDEN_NODES,))
    w2 = W[(INPUT_NODES * HIDDEN_NODES) + HIDDEN_NODES : (INPUT_NODES * HIDDEN_NODES) + HIDDEN_NODES + (HIDDEN_NODES * OUTPUT_NODES)].reshape((HIDDEN_NODES, OUTPUT_NODES))
    b2 = W[(INPUT_NODES * HIDDEN_NODES) + HIDDEN_NODES + (HIDDEN_NODES * OUTPUT_NODES) : (INPUT_NODES * HIDDEN_NODES) + HIDDEN_NODES + (HIDDEN_NODES * OUTPUT_NODES) + OUTPUT_NODES].reshape((OUTPUT_NODES,))

    z1 = np.dot(X, w1) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(a1, w2) + b2
    y_pred = z2.flatten()

    return y_pred

def get_accuracy(Y, Y_pred):
    return r2_score(Y, Y_pred)

def compute_metrics(Y, Y_pred):
    mse = mean_squared_error(Y, Y_pred)
    mae = mean_absolute_error(Y, Y_pred)
    r2 = r2_score(Y, Y_pred)

    msle = np.nan
    if (Y > 0).all() and (Y_pred > 0).all():
        msle = mean_squared_log_error(Y, Y_pred)

    return mse, mae, r2, msle

# Load the dataset
df = pd.read_csv('Bladder.csv')

# Preprocess the dataset
X = df.drop('Post-Bladder volume at the first desire to void', axis=1)
Y = df['Post-Bladder volume at the first desire to void']

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Define the number of nodes in each layer
INPUT_NODES = 13
HIDDEN_NODES = 200
OUTPUT_NODES = 1

no_solution = 100
no_dim = (INPUT_NODES * HIDDEN_NODES) + HIDDEN_NODES + (HIDDEN_NODES * OUTPUT_NODES) + OUTPUT_NODES
w_range = (0.0, 1.0)
lr_range = (0.0, 1.0)
iw_range = (0.9, 0.9)
c = (0.5, 0.3)

s = Swarm(no_solution, no_dim, w_range, lr_range, iw_range, c)

start_time = time.time()
s.optimize(forward_pass, X, Y, 100, 1000)
end_time = time.time()

W = s.get_best_solution()
Y_pred = predict(X, W)

mse, mae, r2, msle = compute_metrics(Y, Y_pred)

print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("R-squared:", r2)
print("Mean Squared Logarithmic Error (MSLE):", msle)

print("Time Consumed: %.3f seconds" % (end_time - start_time))

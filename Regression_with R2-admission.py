#!/usr/bin/env python
# coding: utf-8

# # **Title: GradPredix - Predicting Graduate Admission from Important Parameters**


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import itertools
import random
import os
import csv
get_ipython().system('pip install keras-tuner')
import keras_tuner as kt


# The following code shows a function seed_everything() that sets the seed values for random number generators in TensorFlow, NumPy, and Python's random library, ensuring reproducibility of results for future use cases.

# In[3]:


# Seed Everything to reproduce results for future use cases
def seed_everything(seed=42):
    # Seed value for TensorFlow
    tf.random.set_seed(seed)

    # Seed value for NumPy
    np.random.seed(seed)

    # Seed value for Python's random library
    random.seed(seed)

seed_everything()


# The next code reads in a dataset from a CSV file and assigns it to the variable df. It then displays the first few rows of the dataset using the head() function.

# In[4]:


df = pd.read_csv("Dataset.csv")
df.head()


# In[5]:


df.info()


# 
# This code computes descriptive statistics for each column in the DataFrame and transposes the result.

# In[6]:


df.describe().T


# 
# The code snippet assigns the values of the 'Serial No.' column to the variable 'serialNo'. Then, it drops the 'Serial No.' column from the DataFrame 'df'. Finally, it renames the column 'Chance of Admit ' to 'Chance of Admit'.

# In[7]:


# it may be needed in the future.
serialNo = df["Serial No."].values

df.drop(["Serial No."],axis=1,inplace = True)

df=df.rename(columns = {'Chance of Admit ':'Chance of Admit'})


# **Duplicates and Null values**
# 
# The code prints the number of data points in the dataset. The next code checks for duplicates in the dataset and prints the number of duplicates found. The code then checks for missing values in the dataset by calculating the sum of missing values for each column and prints the results. The code also replaces empty strings with NaN values in the dataset and recalculates the number of missing values.

# In[8]:


# How many reviews do we have?
print('There are', df.shape[0], 'data in this dataset')

# Do we have duplicates?
print('Number of Duplicates:', len(df[df.duplicated()]))

# Do we have missing values?
missing_values = df.isnull().sum()
print('Number of Missing Values by column:\n',missing_values)

print('Number of Missing Values:', df.isnull().sum().sum())

df.replace("", np.nan, inplace=True)
missing_values = df.isnull().sum()
print('Number of Missing Values by column:\n',missing_values)


# **Correlation Matrix**
# 
# A heatmap is generated to visualize the correlation between the features in the DataFrame using the df.corr() function. The annot=True parameter enables displaying the correlation values on the heatmap.

# In[9]:


sns.heatmap(df.corr(), annot=True, cmap='PuOr')


# The correlation matrix of the DataFrame is calculated, and the correlation of "Chance of Admit" with other features is extracted and sorted in descending order.

# In[10]:


# Calculate the correlation matrix
corr_matrix = df.corr()

# Get the correlation of 'price' with other features
corr_with_price = corr_matrix['Chance of Admit']

# Sort these correlations in descending order
sorted_correlations = corr_with_price.sort_values(ascending=False)

# Print out the sorted correlations
sorted_correlations


# **EDA**
# 
# The code sets the style and creates a distribution plot using sns.distplot to visualize the distribution of the "Chance of Admit" column.

# In[11]:


sns.set(style='whitegrid')
f, ax = plt.subplots(1,1, figsize=(12, 8))
ax = sns.histplot(df['Chance of Admit'], kde = True, color = 'c')
plt.title('Distribution of chance of admission')


# 
# The code generates histograms to visualize the distribution of various features in the DataFrame 'df'. Each histogram is created using the seaborn library's 'histplot' function. The features being plotted include 'GRE Score', 'TOEFL Score', 'University Rating', 'SOP Ratings', and 'CGPA'. The 'kde=False' argument indicates that the kernel density estimation plot should not be included. Each histogram is given an appropriate title and displayed using 'plt.show()'. This allows for a quick visual understanding of the distribution of these features in the dataset.

# In[12]:


sns.histplot(df['GRE Score'], kde=False)
plt.title("Distribution of GRE Scores")
plt.show()

sns.histplot(df['TOEFL Score'], kde=False)
plt.title("Distribution of TOEFL Scores")
plt.show()

sns.histplot(df['University Rating'], kde=False)
plt.title("Distribution of University Rating")
plt.show()

sns.histplot(df['SOP'], kde=False)
plt.title("Distribution of SOP Ratings")
plt.show()

sns.histplot(df['CGPA'], kde=False)
plt.title("Distribution of CGPA")
plt.show()


# **Handling Outliers**
# 
# The following code shows the selection of numerical features from a dataframe and the subsequent plotting of boxplots for each of those features. This code is important for visualizing and analyzing the distribution of numerical data, identifying potential outliers in the data.

# In[13]:


import math

# Selecting the numerical features
numerical_features = df.select_dtypes(include=['float64','int64']).columns

rounded_up_value = math.ceil(len(numerical_features)/3)

# Plotting boxplots for all numerical features
fig, axes = plt.subplots(rounded_up_value, 3, figsize=(15, 15))
fig.subplots_adjust(wspace=0.1, hspace=0.3)
fig.suptitle('Outlier Analysis using Boxplot', fontsize=30)
axes = axes.ravel()
font = {
    'family': 'serif',
    'color':  'darkred',
    'weight': 'normal',
    'size': 16,
}
for i, col in enumerate(numerical_features):
    sns.boxplot(df[col], ax=axes[i])
    axes[i].set_title(col, fontdict=font, fontsize=25)


# The next code defines a function called "detect_outliers" that takes a dataframe and a column as input. It calculates the lower and upper limits for outlier detection using the interquartile range (IQR) method and identifies outliers in the specified column. If no outliers are present, it returns an empty list; otherwise, it returns the indices and values of the outliers.

# In[14]:


# Function to detect outliers
def detect_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - (1.5 * IQR)
    upper_limit = Q3 + (1.5 * IQR)
    outliers = df[(df[col] < lower_limit) | (df[col] > upper_limit)]

    if len(outliers) == 0:
        return [], 'No Outliers Present'
    else:
        return outliers.index.tolist(), outliers


# In the following code, all the outlier indices are collected by iterating over the numerical features and applying the "detect_outliers" function. The outliers and their indices are printed for each feature.

# In[15]:


# Collect all outlier indices
all_outlier_indices = []

# Outliers values are --
for col in numerical_features:
    print('-' * 27, col,'-' * 26)
    outlier_indices, outliers = detect_outliers(df, col)
    all_outlier_indices.extend(outlier_indices)
    print(outliers)
    print('\n')


# The code then gets the unique outlier indices by removing any duplicates from the previously collected outlier indices. The unique outlier indices and their count are printed, providing information about the overall number of unique outliers in the dataset.

# In[16]:


# Get unique outlier indices
unique_outlier_indices = list(set(all_outlier_indices))

# Print the unique indices and their count
print("Unique outlier indices:", unique_outlier_indices)
print("Number of unique outlier indices:", len(unique_outlier_indices))


# Next, the code drops the rows with unique outlier indices from the original dataframe, creating a new dataframe called "data" that excludes those outliers.

# In[17]:


df = df.drop(unique_outlier_indices)


# **Feature Selection**
# 
# Following that, the code splits the preprocessed data into input features (X) and target variable (y) by dropping the "price" column from the "data" dataframe.

# In[18]:


X = df.drop('Chance of Admit', axis=1)
y = df['Chance of Admit']


# The subsequent code imports the "train_test_split" function from the "sklearn.model_selection" module. This function is used to split the data into training and testing sets for model evaluation. The data is split such that 80% is used for training and 20% is used for testing, and a random seed of 42 is set for reproducibility.

# In[19]:


from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.20, random_state = 42)


# The code then defines two lists: "num_cols" to store the names of numerical columns (excluding the target variable "Chance of Admit") and "cat_cols" to store the names of categorical columns in the "data" dataframe.

# In[20]:


# Get list of numerical columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Remove 'price' from num_cols
num_cols.remove('Chance of Admit')

# Get list of categorical columns
cat_cols = df.select_dtypes(include=['object']).columns.tolist()

# Print the numerical columns
print("Numerical columns are:", num_cols)

# Print the categorical columns
print("Categorical columns are:", cat_cols)


# In the subsequent code, a column transformer is created using the "make_column_transformer" function from the "sklearn.compose" module. This column transformer applies scaling (MinMaxScaler) to the numerical columns.

# In[21]:


from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# creating column transformer (this will help us normalize/preprocess our data)

ct = make_column_transformer((MinMaxScaler(), num_cols))


# The column transformer is then fitted on the training data, and the same transformation is applied to both the training and test data.

# In[22]:


# fitting column transformer on training data
ct.fit(xtrain)

# transforming training and test data with normalizing (MinMaxScaler) and one hot encoding (OneHotEncoder)
xtrain = ct.transform(xtrain)
xtest = ct.transform(xtest)


# **Model Building**
# 
# Next, the code imports necessary modules from TensorFlow and defines a neural network model called "model_1" using the Sequential API. The model consists of multiple dense layers with dropout regularization to prevent overfitting. The model is compiled with a mean squared error loss function, an Adam optimizer, and metrics to monitor mean absolute error during training.

# In[23]:


from tensorflow.keras import regularizers

# Set random seed
tf.random.set_seed(42)

# Define the model
model_1 = tf.keras.Sequential([
  tf.keras.layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(0.01)),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.01)),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(1)
])

# Define the optimizer
optimizer = tf.keras.optimizers.Adam()

# Compile the model
model_1.compile(loss="mean_squared_error",
                optimizer=optimizer,
                metrics=["mean_absolute_error"])


# A learning rate schedule is defined using a learning rate scheduler callback. This allows the learning rate to change dynamically during training, potentially improving model performance. The model is then trained on the preprocessed training data, using the validation data for evaluation and the learning rate scheduler callback.

# In[24]:


# Define the learning rate schedule
lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-6 * 10**(-epoch / 20))

# Fit the model
history = model_1.fit(xtrain,
                      ytrain,
                      epochs=100,
                      batch_size=32,
                      validation_data=(xtest, ytest),
                      callbacks=[lr_schedule])


# The subsequent code plots the learning rate versus the loss to visualize the relationship between the learning rate and the model's performance. This can help in choosing an appropriate learning rate for training the model.

# In[25]:


# Plot the learning rate versus the loss
lrs = 1e-6 * (10 ** (np.arange(100)/20))
plt.figure(figsize=(10, 7))
plt.semilogx(lrs, history.history["loss"]) # we want the x-axis (learning rate) to be log scale
plt.xlabel("Learning Rate")
plt.ylabel("Loss")
plt.title("Learning rate vs. loss");


# After that, the code resets the TensorFlow session and the weights of the model to ensure a clean state for further experiments or model retraining. A learning rate value (lrx) is also assigned.

# In[26]:


# Reset states generated by Keras
tf.keras.backend.clear_session()

lrx=1e-4
lrx


# The code recompiles the model with the specified learning rate and fits it on the preprocessed training data again, this time using the early stopping callback and model checkpoint.

# In[27]:


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)
ES=EarlyStopping(patience=150, restore_best_weights=True),

# Define the optimizer
optimizer = tf.keras.optimizers.Adam(lr=lrx)

# Compile the model
model_1.compile(loss="mean_squared_error",
                optimizer=optimizer,
                metrics=["mean_absolute_error"])

# Fit the model
history = model_1.fit(xtrain,
                      ytrain,
                      epochs=500,
                      batch_size=32,
                      validation_data=(xtest, ytest),
                      callbacks=[ES,checkpoint])


# A function called "plot_losses" is defined to plot the losses (training and validation) from the history object returned by the model training process.

# In[28]:


def plot_losses(history):
    losses = pd.DataFrame(history.history)
    losses.plot()


# In[29]:


plot_losses(history)


# The next code loads the best saved model from the model checkpoint file ("best_model.h5").

# In[30]:


from tensorflow.keras.models import load_model

# Load the best saved model
model = load_model('best_model.h5')


# In[31]:


model.evaluate(xtest, ytest)


# In[32]:


# Predictions
testPredict = model.predict(xtest)


# Test predictions are made using the loaded model, and an evaluation function is defined to calculate and print the mean absolute error (MAE) and mean squared error (MSE) between the true target values (ytest) and the predicted values (testPredict).

# Model Evaluation
from sklearn.metrics import r2_score

def evaluate_model(y_true, y_pred):
    mae = tf.keras.metrics.mean_absolute_error(y_true, y_pred)
    mse = tf.keras.metrics.mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print('Mean Absolute Error: ', mae.numpy())
    print('Mean Squared Error: ', mse.numpy())
    print('R-squared (R2) Score: ', r2)


# The code then evaluates the model's performance by calling the evaluation function with the true target values and the predicted values. The MAE and MSE are printed, providing an assessment of the model's predictive accuracy.

# In[34]:


evaluate_model(ytest, tf.squeeze(testPredict))  # Flatten the prediction array to match the shape


# **Hyperparameter Tuning**
# 
# The following code defines a function called "build_model" that constructs a neural network model with customizable hyperparameters. The model includes multiple layers, dropout regularization, and batch normalization. It is compiled with an Adam optimizer and mean squared error loss.

# In[35]:


def build_model(hp):
    from keras.layers import BatchNormalization
    from keras.regularizers import l1_l2

    # Define all hyperparameters
    n_layers = hp.Choice('n_layers', [2, 4, 6])
    dropout_rate = hp.Choice('rate', [0.2, 0.4, 0.5, 0.7])
    n_units = hp.Choice('units', [64, 128, 256, 512])
    l1_reg = hp.Choice('l1', [0.0, 0.01, 0.001, 0.0001])
    l2_reg = hp.Choice('l2', [0.0, 0.01, 0.001, 0.0001])

    # Model architecture
    model = Sequential()

    # Input layer
    model.add(Dense(n_units, input_dim=xtrain.shape[1], activation='relu', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)))
    model.add(BatchNormalization())

    # Add hidden layers
    for _ in range(n_layers):
        model.add(Dense(n_units, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)))
        model.add(BatchNormalization())

    # Add Dropout Layer
    model.add(Dropout(dropout_rate))

    # Output Layer
    model.add(Dense(1))

    # Define the optimizer
    optimizer = tf.keras.optimizers.Adam(lr=lrx)

    # Compile the model
    model.compile(loss="mean_squared_error",
                    optimizer=optimizer,
                    metrics=["mean_absolute_error"])

    # Return model
    return model


# In the next code block, a random search is initialized to find the best model architecture. The search explores different hyperparameter combinations and aims to minimize validation loss. This approach helps in identifying the optimal model configuration.

# In[36]:


# Initialize Random Searcher
random_searcher = kt.RandomSearch(
    hypermodel=build_model,
    objective='val_loss',
    max_trials=30,
    seed=42,
    project_name="Pgt-Search"
)

# Start Searching
search = random_searcher.search(
    xtrain, ytrain,
    validation_data=(xtest, ytest),
    epochs = 30,
    batch_size = 64
)


# In[37]:


BATCH_SIZE=64


# The code then imports necessary modules for callbacks and collects the best model architecture from the random search. The model is compiled and trained on the preprocessed data, with early stopping and model checkpoint implemented to prevent overfitting and save the best model based on validation loss.

# In[38]:


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Collect the best model Architecture obtained by Random Searcher
best_model = build_model(random_searcher.get_best_hyperparameters(num_trials=1)[0])

# Model Architecture
best_model.summary()

optimizer = tf.keras.optimizers.Adam(lr=lrx)

# Compile Model
best_model.compile(loss="mean_squared_error",
                    optimizer=optimizer,
                    metrics=["mean_absolute_error"])
# Model Training
best_model_history = best_model.fit(
    xtrain, ytrain,
    validation_data=(xtest, ytest),
    epochs = 500,
    batch_size = BATCH_SIZE*2,
    callbacks = [
  EarlyStopping(patience=150, restore_best_weights=True),
  ModelCheckpoint('best-model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)]
)


# After training, the best model is loaded and its architecture summary is printed.

# In[39]:


#  Load model
best_model = tf.keras.models.load_model('best-model.h5')
best_model.summary()


# Next, the best model's performance is evaluated by calculating mean squared error and mean absolute error on the test data. This assessment provides insights into the model's predictive accuracy.

# In[40]:


# Evaluate the best model
best_test_loss, best_test_acc = best_model.evaluate(xtest, ytest)
print(f"Validation loss (MSE) after Tuning     : {best_test_loss} ")
print(f"Validation MAE after Tuning : {best_test_acc}  ")


# The subsequent code extracts training and validation metrics from the best model's training history. These metrics, including mean absolute error and loss, are plotted over epochs. This visualization helps in understanding the model's learning progress.

# In[41]:


# Extract the history from the best model
accuracy = best_model_history.history['mean_absolute_error']
val_accuracy = best_model_history.history['val_mean_absolute_error']

loss = best_model_history.history['loss']
val_loss = best_model_history.history['val_loss']

epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'b', label='Training mean_absolute_error')
plt.plot(epochs, val_accuracy, 'r', label='Validation mean_absolute_error')

plt.title('Training and validation mean_absolute_error')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')

plt.title('Training and validation loss')
plt.legend()
plt.show()


# A function called "make_sample_prediction" is defined to make predictions on individual test samples using the best model. It calculates differences and percentage differences between true and predicted prices.

# In[42]:


def make_sample_prediction(index, xtest, ytest, model, transformed_cols, original_data):
    # Select the corresponding test sample and true price
    sample, true_price = xtest[index], ytest.iloc[index]
    original_sample = original_data.iloc[index]

    # Reshape the sample to be a batch of size 1
    sample = sample.reshape(1, -1)

    # Make a prediction
    predicted_price = model.predict(sample)[0][0]

    # Calculate the difference
    difference = true_price - predicted_price

    # Calculate the percentage difference
    pct_difference = (difference / true_price) * 100

    # Create a data frame for the scaled sample
    sample_df = pd.DataFrame(sample, columns=transformed_cols)

    # Print out the sample, the true price, and the predicted price
    print(f"Prediction {index+1}")
    print("Original Features:")
    print(original_sample)
#     print("Features (scaled):")
    sample_df
    print("True Price: ", true_price)
    print("Predicted Price: ", predicted_price)
    print("Difference: ", difference)
    print("Percentage Difference: ", pct_difference, "%")
    print("\n---------------------\n")

transformed_cols = ct.transformers_[0][2]


# Then, the code calls the "make_sample_prediction" function for the first 10 samples in the test data, providing predictions, true prices, differences, and percentage differences. This step aids in understanding the model's predictive capabilities on specific samples.

# In[43]:


# Call the function for the first 10 samples
for i in range(10):
    make_sample_prediction(i, xtest, ytest, best_model, transformed_cols, df)


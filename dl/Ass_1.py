#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns


# In[6]:


# Load the dataset
data = pd.read_csv('1_boston_housing.csv')


# In[7]:


# Assume the target column is named 'MEDV'
X = data.drop("MEDV", axis=1)
Y = data["MEDV"]


# In[8]:


data.head()


# In[9]:


data.shape


# In[10]:


data.describe()


# In[11]:


# Preprocess the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[12]:


# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)


# In[13]:


# Build the model
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))  # Output layer for regression


# In[14]:


# Compile the model
model.compile(loss='mse', optimizer='adam', metrics=['mae'])


# In[15]:


# Train the model
history = model.fit(X_train, Y_train, epochs=10, batch_size=1, verbose=1, validation_data=(X_test, Y_test))


# In[16]:


# Evalute the model
mse = model.evaluate(X_test, Y_test)
print("Mean Squared Error:", mse)


# In[17]:


# Predictions
y_pred = model.predict(X_test)
print(y_pred[:5])


# In[18]:


# Visualizing Predicted vs Actual Pricesplt.figure(figsize=(10, 6))
plt.scatter(Y_test, y_pred, alpha=0.6)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--', lw=2)
plt.title('Predicted vs Actual Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.xlim([0, 60])
plt.ylim([0, 60])
plt.grid()
plt.show()


# In[ ]:





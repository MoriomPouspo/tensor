# import necessary modules

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# load dataset
df = pd.read_csv('monthly_milk_production.csv',
				index_col='Date',
				parse_dates=True)
df.index.freq = 'MS'

# testing if dataset is loaded
print(df.head())


#Now proceed to perform EDA analysis on the dataset.
# Plotting graph b/w production and date
df.plot(figsize=(12, 6))
plt.show() #extraa


# seasonal analysis of time series data
from statsmodels.tsa.seasonal import seasonal_decompose
results = seasonal_decompose(df['Production'])
results.plot()
plt.show() #extra

#spliting the data into training and testng
train = df.iloc[:156]
test = df.iloc[156:]

#Scaling our data to perform computations in a fast and accurate manner.
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)

#Processing to a time series generation.

from keras.preprocessing.sequence import TimeseriesGenerator
 
n_input = 3
n_features = 1
generator = TimeseriesGenerator(scaled_train,
                                scaled_train,
                                length=n_input,
                                batch_size=1)
X, y = generator[0]
print(f'Given the Array: \n{X.flatten()}')
print(f'Predict this y: \n {y}')
# We do the same thing, but now instead for 12 months
n_input = 12
generator = TimeseriesGenerator(scaled_train,
                                scaled_train,
                                length=n_input,
                                batch_size=1)

# Now let’s define the architecture of the model using TensorFlow API.

# define model
model = Sequential()
model.add(LSTM(100, activation='relu',
               input_shape=(n_input, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
#print(loss)
model.summary()
model.fit(generator, epochs=5)





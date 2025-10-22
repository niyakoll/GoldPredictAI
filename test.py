import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import openpyxl
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

"""
pip install yfinance pandas scikit-learn numpy openpyxl tensorflow matplotlib
"""
# Fetch gold data (e.g., last 5 years)
#gold_data = yf.download('GC=F', start='2018-01-01', end='2019-01-01', actions=True)
#gold_data.to_csv('gold_prices.csv')  # Save for later use

# Load and inspect
#df = pd.read_csv('gold_prices.csv')
df = pd.read_excel("data.xlsx","data")
print(df.head())




# Assume df has 'Close' as target
data = df['Close'].values.reshape(-1, 1)

# Scale
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create sequences
def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 60
X, y = create_sequences(scaled_data, seq_length)

# Split (chronological)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Reshape for RNN: (samples, timesteps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=25))
model.add(Dense(units=1))  # Output: predicted price

# Compile
model.compile(optimizer='adam', loss='mean_squared_error')

# Train
history = model.fit(X_train, y_train, batch_size=32, epochs=80, validation_data=(X_test, y_test))
# Predict
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)  # Unscale
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# Metrics
rmse = np.sqrt(mean_squared_error(y_test_inv, predictions))
print(f'RMSE: {rmse}')

# Plot
plt.plot(y_test_inv, label='Actual')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.show()

#save model
model.save('C:/Users/240144419/GoldPredictAI/gold_rnn_model.h5')
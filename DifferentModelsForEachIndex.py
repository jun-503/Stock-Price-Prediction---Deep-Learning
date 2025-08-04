import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import mixed_precision
import matplotlib.pyplot as plt

# Enable mixed precision (optional but faster on GPU)
mixed_precision.set_global_policy('mixed_float16')

df = pd.read_csv('C:\\Users\\junai\\Documents\\ML Projects\\Stock prediction\Data\\indexProcessed.csv')  # if not already loaded

# Make sure Date is datetime
df['Date'] = pd.to_datetime(df['Date'])

# Sliding window creator
def create_sliding_window_data(series, window_size):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i + window_size])
        y.append(series[i + window_size])
    return np.array(X), np.array(y)

# Create folder to save models
os.makedirs("saved_models", exist_ok=True)

# Parameters
window_size = 50
results = {}

for index_name in df['Index'].unique():
    print(f" Training model for: {index_name}")
    df_index = df[df['Index'] == index_name].sort_values('Date').copy()

    # Skip very short series
    if len(df_index) < window_size + 100:
        print(f" Skipping {index_name} (too few samples)")
        continue

    # Scale CloseUSD
    scaler = MinMaxScaler()
    df_index['CloseUSD_scaled'] = scaler.fit_transform(df_index[['CloseUSD']])

    # Create sliding windows
    series = df_index['CloseUSD_scaled'].values
    X, y = create_sliding_window_data(series, window_size)
    X = X.reshape(-1, window_size, 1) # -1 means infer the number of samples automatically, window_size is the number of time steps, and 1 is the number of features (univariate time series)

    # Split (no shuffle for time series)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Define model
    model = Sequential([
        LSTM(32, return_sequences=False, input_shape=(window_size, 1)),
        Dropout(0.1),
        Dense(16, activation='relu'),
        Dense(1)  # output layer
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=64,
        validation_data=(X_test, y_test),
        callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
        verbose=1
    )

    # Evaluate
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    results[index_name] = test_loss
    print(f" {index_name} Test Loss: {test_loss:.5f}")

    # Save model
    model.save(f"saved_models/{index_name}_lstm_model.h5")

    # Plot optional
    plt.figure(figsize=(10, 4))
    preds = model.predict(X_test)
    plt.plot(scaler.inverse_transform(y_test.reshape(-1, 1)), label='Actual', color='blue')
    plt.plot(scaler.inverse_transform(preds), label='Predicted', color='red')
    plt.title(f'{index_name} - Actual vs Predicted')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

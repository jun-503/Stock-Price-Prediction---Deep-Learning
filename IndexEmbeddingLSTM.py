import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Embedding, RepeatVector, Concatenate,Lambda
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau

import matplotlib.pyplot as plt




# ------------------------------
# Load and preprocess data
# ------------------------------
df = pd.read_csv('C:\\Users\\junai\\Documents\\ML Projects\\Stock prediction\Data\\indexProcessed_ta.csv')  # Replace with your actual file path

# Convert Date to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Encode Index labels to integers
df['IndexID'] = LabelEncoder().fit_transform(df['Index'])

# Sort to maintain temporal order
df = df.sort_values(['Index', 'Date']).reset_index(drop=True) # Ensure the data is sorted by Index and then within each Index by Date and reset the index also drop the old index

# Parameters
window_size = 35
features = ['others_cr', 'CloseUSD', 'trend_ichimoku_base', 'trend_ichimoku_conv', 'momentum_kama'] # List of features to use

X_all, y_all, index_ids = [], [], []

# ------------------------------
# Create sliding window sequences per index
# ------------------------------
for idx in df['Index'].unique():
    sub_df = df[df['Index'] == idx].copy()
    
    # scaling the features
    scaler = MinMaxScaler()
    sub_df[features] = scaler.fit_transform(sub_df[features])

    series = sub_df[features].values
    index_id = sub_df['IndexID'].iloc[0]

    for i in range(len(series) - window_size):
        X_all.append(series[i:i + window_size])
        y_all.append(series[i + window_size][features.index('CloseUSD')])  # scaled CloseUSD

        index_ids.append(index_id)


# Convert to arrays
X_all = np.array(X_all).reshape(-1, window_size, len(features)) # -1 means infer the number of samples automatically, window_size is the number of time steps, and 1 is the number of features (univariate time series)
y_all = np.array(y_all) # Target variable
index_ids = np.array(index_ids)

# ------------------------------
# Train-test split (no shuffle)
# ------------------------------
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_all, y_all, index_ids, test_size=0.2, shuffle=False
)

# ------------------------------
# Build LSTM model with Index Embedding
# ------------------------------
num_indices = len(np.unique(index_ids))

seq_input = Input(shape=(window_size, len(features)), name='sequence_input')
idx_input = Input(shape=(1,), name='index_input')

# Embedding layer for index
embedding = Embedding(input_dim=num_indices, output_dim=8)(idx_input) # Embedding layer for IndexID
# Reshape embedding to match sequence length
embedding = Lambda(lambda x: tf.squeeze(x, axis=1))(embedding)  # shape: (None, 8)
embedding = RepeatVector(window_size)(embedding)  # shape: (None, 50, 8)

# Concatenate sequence input with index embedding
merged = Concatenate(axis=-1)([seq_input, embedding])

# Wrong Model Type: You used Sequential which can only handle single input, but you need multiple inputs (sequence + index)
# LSTM outputs 3D tensor, so we need to use return_sequences=False i.e. only return the last output/last time step of the LSTM (32,128)
x = LSTM(128, return_sequences=False)(merged)
x = Dropout(0.2)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(32, activation='relu')(x)
output = Dense(1, dtype='float32')(x)  # Fixed: output matches number of features

model = Model(inputs=[seq_input, idx_input], outputs=output)





model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# ------------------------------
# Train the model
# ------------------------------
early_stop = EarlyStopping(patience=3, 
restore_best_weights=True,
monitor='val_loss',
verbose=1)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',      # Watch the validation loss metric
    factor=0.5,              # Reduce learning rate by half when triggered
    patience=3,              # Wait 3 epochs before reducing LR
    min_lr=1e-7,             # Don't go below this minimum learning rate
    verbose=1                # Print messages when LR is reduced
)
history = model.fit(
    [X_train, idx_train], y_train,
    validation_data=([X_test, idx_test], y_test),
    epochs=5,
    batch_size=64,
    callbacks=[early_stop, reduce_lr],  # Fixed: added reduce_lr callback
    verbose=1
)

# ------------------------------
# Evaluate and Plot
# ------------------------------
test_loss = model.evaluate([X_test, idx_test], y_test)
print(f'Test Loss: {test_loss:.5f}')

preds = model.predict([X_test, idx_test])

plt.figure(figsize=(14, 6))
plt.plot(y_test[:500], label='Actual', color='blue')
plt.plot(preds[:500], label='Predicted', color='red')
plt.title('Actual vs Predicted Closing Price (first 500 test points)')
plt.xlabel('Sample')
plt.ylabel('Scaled CloseUSD')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------------------
# Save model
# ------------------------------
model.save("index_aware_lstm_model.h5")

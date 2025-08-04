import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Embedding, RepeatVector, Concatenate, Lambda, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# ------------------------------
# Load and preprocess data
# ------------------------------
df = pd.read_csv('Data/indexProcessed.csv')
df['Date'] = pd.to_datetime(df['Date'])
df['IndexID'] = LabelEncoder().fit_transform(df['Index'])
df = df.sort_values(['Index', 'Date']).reset_index(drop=True)

# ------------------------------
# Quick Feature Engineering (Most Important)
# ------------------------------
def add_basic_features(df):
    """Add the most important technical indicators"""
    df = df.copy()
    
    # Returns
    df['Returns'] = df.groupby('Index')['CloseUSD'].pct_change()
    
    # Moving averages
    df['MA_5'] = df.groupby('Index')['CloseUSD'].rolling(window=5).mean().reset_index(0, drop=True)
    df['MA_20'] = df.groupby('Index')['CloseUSD'].rolling(window=20).mean().reset_index(0, drop=True)
    
    # Price ratios
    df['Price_to_MA5'] = df['CloseUSD'] / df['MA_5']
    df['Price_to_MA20'] = df['CloseUSD'] / df['MA_20']
    
    # Volatility
    df['Volatility_5'] = df.groupby('Index')['Returns'].rolling(window=5).std().reset_index(0, drop=True)
    
    # Volume ratio
    df['Volume_MA_5'] = df.groupby('Index')['Volume'].rolling(window=5).mean().reset_index(0, drop=True)
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_5']
    
    return df

df = add_basic_features(df)
df = df.dropna()

# ------------------------------
# Enhanced Parameters
# ------------------------------
window_size = 60  # Increased from 50
feature_columns = ['CloseUSD', 'Volume', 'Open', 'Returns', 'MA_5', 'MA_20', 
                   'Price_to_MA5', 'Price_to_MA20', 'Volatility_5', 'Volume_Ratio']

X_all, y_all, index_ids = [], [], []

# ------------------------------
# Create sequences with multiple features
# ------------------------------
for idx in df['Index'].unique():
    sub_df = df[df['Index'] == idx].copy()
    
    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(sub_df[feature_columns])
    
    # Target
    target_scaler = StandardScaler()
    target_scaled = target_scaler.fit_transform(sub_df[['CloseUSD']])
    
    index_id = sub_df['IndexID'].iloc[0]

    for i in range(len(features_scaled) - window_size):
        X_all.append(features_scaled[i:i + window_size])
        y_all.append(target_scaled[i + window_size, 0])
        index_ids.append(index_id)

X_all = np.array(X_all)
y_all = np.array(y_all)
index_ids = np.array(index_ids)

# ------------------------------
# Temporal train-test split
# ------------------------------
split_idx = int(len(X_all) * 0.8)
X_train, X_test = X_all[:split_idx], X_all[split_idx:]
y_train, y_test = y_all[:split_idx], y_all[split_idx:]
idx_train, idx_test = index_ids[:split_idx], index_ids[split_idx:]

# ------------------------------
# Improved Model Architecture
# ------------------------------
num_indices = len(np.unique(index_ids))
num_features = X_train.shape[2]

def create_improved_model(window_size, num_features, num_indices):
    seq_input = Input(shape=(window_size, num_features))
    idx_input = Input(shape=(1,))
    
    # Enhanced embedding
    embedding = Embedding(input_dim=num_indices, output_dim=16)(idx_input)
    embedding = Lambda(lambda x: tf.squeeze(x, axis=1))(embedding)
    embedding = RepeatVector(window_size)(embedding)
    
    # Concatenate
    merged = Concatenate(axis=-1)([seq_input, embedding])
    
    # Bidirectional LSTM layers
    lstm1 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(merged)
    lstm2 = Bidirectional(LSTM(64, return_sequences=False, dropout=0.2))(lstm1)
    
    # Dense layers with better regularization
    dense1 = Dense(64, activation='relu')(lstm2)
    dense1 = Dropout(0.3)(dense1)
    
    dense2 = Dense(32, activation='relu')(dense1)
    dense2 = Dropout(0.2)(dense2)
    
    output = Dense(1, activation='linear')(dense2)
    
    model = Model(inputs=[seq_input, idx_input], outputs=output)
    return model

model = create_improved_model(window_size, num_features, num_indices)

# Better optimizer
optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

# ------------------------------
# Enhanced Training
# ------------------------------
early_stop = EarlyStopping(patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-7)

history = model.fit(
    [X_train, idx_train], y_train,
    validation_data=([X_test, idx_test], y_test),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# ------------------------------
# Evaluation
# ------------------------------
preds = model.predict([X_test, idx_test])

mse = mean_squared_error(y_test, preds)
mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, preds)

# Directional accuracy
actual_direction = np.diff(y_test) > 0
pred_direction = np.diff(preds.flatten()) > 0
directional_accuracy = np.mean(actual_direction == pred_direction)

print(f"Test MSE: {mse:.6f}")
print(f"Test MAE: {mae:.6f}")
print(f"Test RMSE: {rmse:.6f}")
print(f"Test R²: {r2:.6f}")
print(f"Directional Accuracy: {directional_accuracy:.4f}")

# ------------------------------
# Visualization
# ------------------------------
plt.figure(figsize=(15, 10))

# Training history
plt.subplot(2, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Predictions
plt.subplot(2, 2, 2)
sample_size = min(500, len(y_test))
plt.plot(y_test[:sample_size], label='Actual', alpha=0.7)
plt.plot(preds[:sample_size], label='Predicted', alpha=0.7)
plt.title('Actual vs Predicted')
plt.xlabel('Sample')
plt.ylabel('Scaled Price')
plt.legend()
plt.grid(True)

# Scatter plot
plt.subplot(2, 2, 3)
plt.scatter(y_test, preds, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Scatter Plot')
plt.grid(True)

# MAE history
plt.subplot(2, 2, 4)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('quick_improvements_results.png', dpi=300, bbox_inches='tight')
plt.show()

# Save model
model.save("quick_improved_model.h5")

print("\nQuick Improvements Summary:")
print("1. Added 10 technical indicators")
print("2. Increased window size to 60")
print("3. Used StandardScaler instead of MinMaxScaler")
print("4. Added Bidirectional LSTM layers")
print("5. Enhanced training with learning rate scheduling")
print("6. Better evaluation metrics")
print(f"Final MSE: {mse:.6f}")
print(f"Directional Accuracy: {directional_accuracy:.4f}") 
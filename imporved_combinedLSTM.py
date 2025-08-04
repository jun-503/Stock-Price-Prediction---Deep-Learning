import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Embedding, RepeatVector, Concatenate, Lambda, Bidirectional, BatchNormalization, Attention
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import mixed_precision
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# ------------------------------
# Enable mixed precision (optional, speeds up on GPU)
# ------------------------------
mixed_precision.set_global_policy('mixed_float16')

# ------------------------------
# Load and preprocess data
# ------------------------------
df = pd.read_csv('Data/indexProcessed.csv')

# Convert Date to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Encode Index labels to integers
df['IndexID'] = LabelEncoder().fit_transform(df['Index'])

# Sort to maintain temporal order
df = df.sort_values(['Index', 'Date']).reset_index(drop=True)

# ------------------------------
# Enhanced Feature Engineering
# ------------------------------
def create_technical_indicators(df):
    """Create technical indicators for better feature engineering"""
    df = df.copy()
    
    # Price-based features
    df['Returns'] = df.groupby('Index')['CloseUSD'].pct_change()
    df['Log_Returns'] = df.groupby('Index')['CloseUSD'].apply(lambda x: np.log(x / x.shift(1)))
    
    # Moving averages
    df['MA_5'] = df.groupby('Index')['CloseUSD'].rolling(window=5).mean().reset_index(0, drop=True)
    df['MA_20'] = df.groupby('Index')['CloseUSD'].rolling(window=20).mean().reset_index(0, drop=True)
    df['MA_50'] = df.groupby('Index')['CloseUSD'].rolling(window=50).mean().reset_index(0, drop=True)
    
    # Price ratios
    df['Price_to_MA5'] = df['CloseUSD'] / df['MA_5']
    df['Price_to_MA20'] = df['CloseUSD'] / df['MA_20']
    df['Price_to_MA50'] = df['CloseUSD'] / df['MA_50']
    
    # Volatility
    df['Volatility_5'] = df.groupby('Index')['Returns'].rolling(window=5).std().reset_index(0, drop=True)
    df['Volatility_20'] = df.groupby('Index')['Returns'].rolling(window=20).std().reset_index(0, drop=True)
    
    # RSI (Relative Strength Index)
    def calculate_rsi(prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    df['RSI'] = df.groupby('Index')['CloseUSD'].apply(calculate_rsi).reset_index(0, drop=True)
    
    # MACD
    exp1 = df.groupby('Index')['CloseUSD'].ewm(span=12).mean()
    exp2 = df.groupby('Index')['CloseUSD'].ewm(span=26).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df.groupby('Index')['MACD'].ewm(span=9).mean().reset_index(0, drop=True)
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # Bollinger Bands
    df['BB_Middle'] = df.groupby('Index')['CloseUSD'].rolling(window=20).mean().reset_index(0, drop=True)
    bb_std = df.groupby('Index')['CloseUSD'].rolling(window=20).std().reset_index(0, drop=True)
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    df['BB_Position'] = (df['CloseUSD'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    # Volume features
    df['Volume_MA_5'] = df.groupby('Index')['Volume'].rolling(window=5).mean().reset_index(0, drop=True)
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_5']
    
    # Time-based features
    df['Day_of_Week'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    df['Year'] = df['Date'].dt.year
    
    return df

# Apply feature engineering
df = create_technical_indicators(df)

# ------------------------------
# Feature Selection and Scaling
# ------------------------------
# Select features for the model
feature_columns = [
    'CloseUSD', 'Volume', 'Open', 'High', 'Low',
    'Returns', 'Log_Returns', 'MA_5', 'MA_20', 'MA_50',
    'Price_to_MA5', 'Price_to_MA20', 'Price_to_MA50',
    'Volatility_5', 'Volatility_20', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram',
    'BB_Position', 'Volume_Ratio', 'Day_of_Week', 'Month', 'Quarter'
]

# Remove rows with NaN values (from rolling calculations)
df = df.dropna()

# Parameters
window_size = 60  # Increased window size for better pattern recognition
X_all, y_all, index_ids = [], [], []

# ------------------------------
# Create sliding window sequences per index with multiple features
# ------------------------------
for idx in df['Index'].unique():
    sub_df = df[df['Index'] == idx].copy()
    
    # Scale features
    scaler = StandardScaler()  # Using StandardScaler instead of MinMaxScaler
    features_scaled = scaler.fit_transform(sub_df[feature_columns])
    
    # Create feature matrix
    feature_matrix = pd.DataFrame(features_scaled, columns=feature_columns, index=sub_df.index)
    
    # Target variable (next day's close price)
    target_scaler = StandardScaler()
    target_scaled = target_scaler.fit_transform(sub_df[['CloseUSD']])
    
    index_id = sub_df['IndexID'].iloc[0]

    for i in range(len(feature_matrix) - window_size):
        X_all.append(feature_matrix.iloc[i:i + window_size].values)
        y_all.append(target_scaled[i + window_size, 0])  # Next day's scaled close price
        index_ids.append(index_id)

# Convert to arrays
X_all = np.array(X_all)  # Shape: (samples, window_size, features)
y_all = np.array(y_all)
index_ids = np.array(index_ids)

print(f"Dataset shape: {X_all.shape}")
print(f"Number of features: {X_all.shape[2]}")
print(f"Number of indices: {len(np.unique(index_ids))}")

# ------------------------------
# Train-test split (temporal split)
# ------------------------------
# Use last 20% of data for testing (temporal split)
split_idx = int(len(X_all) * 0.8)
X_train, X_test = X_all[:split_idx], X_all[split_idx:]
y_train, y_test = y_all[:split_idx], y_all[split_idx:]
idx_train, idx_test = index_ids[:split_idx], index_ids[split_idx:]

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# ------------------------------
# Enhanced LSTM model with Index Embedding
# ------------------------------
num_indices = len(np.unique(index_ids))
num_features = X_train.shape[2]

def create_improved_model(window_size, num_features, num_indices):
    """Create an improved LSTM model with attention mechanism"""
    
    # Input layers
    seq_input = Input(shape=(window_size, num_features), name='sequence_input')
    idx_input = Input(shape=(1,), name='index_input')
    
    # Enhanced embedding layer for index
    embedding = Embedding(input_dim=num_indices, output_dim=16)(idx_input)
    embedding = Lambda(lambda x: tf.squeeze(x, axis=1))(embedding)
    embedding = RepeatVector(window_size)(embedding)
    
    # Concatenate sequence input with index embedding
    merged = Concatenate(axis=-1)([seq_input, embedding])
    
    # First LSTM layer with return sequences
    lstm1 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(merged)
    lstm1 = BatchNormalization()(lstm1)
    
    # Second LSTM layer
    lstm2 = Bidirectional(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(lstm1)
    lstm2 = BatchNormalization()(lstm2)
    
    # Third LSTM layer (final)
    lstm3 = Bidirectional(LSTM(32, return_sequences=False, dropout=0.2, recurrent_dropout=0.2))(lstm2)
    lstm3 = BatchNormalization()(lstm3)
    
    # Dense layers with regularization
    dense1 = Dense(64, activation='relu')(lstm3)
    dense1 = Dropout(0.3)(dense1)
    dense1 = BatchNormalization()(dense1)
    
    dense2 = Dense(32, activation='relu')(dense1)
    dense2 = Dropout(0.2)(dense2)
    dense2 = BatchNormalization()(dense2)
    
    # Output layer
    output = Dense(1, activation='linear', dtype='float32')(dense2)
    
    model = Model(inputs=[seq_input, idx_input], outputs=output)
    return model

# Create model
model = create_improved_model(window_size, num_features, num_indices)

# Compile with better optimizer settings
optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

model.summary()

# ------------------------------
# Enhanced Training with Callbacks
# ------------------------------
# Callbacks for better training
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

checkpoint = ModelCheckpoint(
    'best_model.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# Train the model
history = model.fit(
    [X_train, idx_train], y_train,
    validation_data=([X_test, idx_test], y_test),
    epochs=100,
    batch_size=32,  # Smaller batch size for better generalization
    callbacks=[early_stop, reduce_lr, checkpoint],
    verbose=1
)

# ------------------------------
# Enhanced Evaluation and Metrics
# ------------------------------
def evaluate_model(model, X_test, y_test, idx_test):
    """Comprehensive model evaluation"""
    
    # Predictions
    preds = model.predict([X_test, idx_test])
    
    # Calculate metrics
    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)
    
    # Calculate directional accuracy
    actual_direction = np.diff(y_test) > 0
    pred_direction = np.diff(preds.flatten()) > 0
    directional_accuracy = np.mean(actual_direction == pred_direction)
    
    print(f"Test MSE: {mse:.6f}")
    print(f"Test MAE: {mae:.6f}")
    print(f"Test RMSE: {rmse:.6f}")
    print(f"Test R²: {r2:.6f}")
    print(f"Directional Accuracy: {directional_accuracy:.4f}")
    
    return preds, {'mse': mse, 'mae': mae, 'rmse': rmse, 'r2': r2, 'directional_accuracy': directional_accuracy}

# Evaluate model
preds, metrics = evaluate_model(model, X_test, y_test, idx_test)

# ------------------------------
# Enhanced Visualization
# ------------------------------
def plot_training_history(history):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # MAE plot
    ax2.plot(history.history['mae'], label='Training MAE')
    ax2.plot(history.history['val_mae'], label='Validation MAE')
    ax2.set_title('Model MAE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_predictions(y_test, preds, sample_size=1000):
    """Plot actual vs predicted values"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Time series plot
    sample_size = min(sample_size, len(y_test))
    ax1.plot(y_test[:sample_size], label='Actual', color='blue', alpha=0.7)
    ax1.plot(preds[:sample_size], label='Predicted', color='red', alpha=0.7)
    ax1.set_title('Actual vs Predicted Values (First {} samples)'.format(sample_size))
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Scaled Close Price')
    ax1.legend()
    ax1.grid(True)
    
    # Scatter plot
    ax2.scatter(y_test, preds, alpha=0.5)
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax2.set_xlabel('Actual Values')
    ax2.set_ylabel('Predicted Values')
    ax2.set_title('Actual vs Predicted Scatter Plot')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# Plot results
plot_training_history(history)
plot_predictions(y_test, preds)

# ------------------------------
# Save model and results
# ------------------------------
model.save("improved_index_aware_lstm_model.h5")

# Save metrics
import json
with open('model_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

print("Model and metrics saved successfully!")
print(f"Final model metrics: {metrics}")
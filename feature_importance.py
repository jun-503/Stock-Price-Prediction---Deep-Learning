import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ------------------------------
# Load and preprocess data
# ------------------------------
df = pd.read_csv('Data/indexProcessed_ta.csv')

# Drop raw price columns
df.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'], inplace=True)

# Filter only HSI index
df = df[df['Index'] == 'HSI'].copy()

# Add engineered features before dropping 'Date'
df['Date'] = pd.to_datetime(df['Date'])
df['Day_of_Week'] = df['Date'].dt.dayofweek
df['Month'] = df['Date'].dt.month
df['Quarter'] = df['Date'].dt.quarter

# Example returns and ratios
df['Returns'] = df['CloseUSD'].pct_change()  # returns means the percentage change of the close price
df['Log_Returns'] = np.log(df['CloseUSD'] / df['CloseUSD'].shift(1)) # log returns means the log of the percentage change of the close price

# Moving averages
for ma in [5, 20, 50]:
    df[f'MA_{ma}'] = df['CloseUSD'].rolling(ma).mean()   # .rolling calculate the moving average
    df[f'Price_to_MA{ma}'] = df['CloseUSD'] / df[f'MA_{ma}']
    df[f'Volatility_{ma}'] = df['CloseUSD'].rolling(ma).std()

# Volume Ratio
# We already dropped 'Volume', so you may want to reload it from original if needed
# Otherwise comment this out or adapt if you plan to include volume again
# df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(5).mean()

# Drop unnecessary columns again
df.drop(columns=['Index', 'Date'], inplace=True)

# Handle missing values
df.dropna(axis=1, thresh=int(0.9 * len(df)), inplace=True)  # drop features with >10% missing
df.fillna(method='ffill', inplace=True)

print("1")
# ------------------------------
# Standardize features
# ------------------------------
scaler = StandardScaler()
scaled_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

print("2")
# ------------------------------
# Create sliding windows
# ------------------------------
window_size = 15  # Reduced for efficiency
X = []
y = []

for i in range(window_size, len(scaled_df) - 1):
    X.append(scaled_df.iloc[i - window_size:i].values.flatten())
    y.append(scaled_df.iloc[i]['CloseUSD'])

X = np.array(X)
y = np.array(y)

print("3")
# ------------------------------
# Train-test split
# ------------------------------
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, shuffle=False)

print("4")
# ------------------------------
# Train Random Forest
# ------------------------------
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

print("5")
# ------------------------------
# Feature importance analysis
# ------------------------------
n_features = scaled_df.shape[1]
feature_names = scaled_df.columns

importances = rf.feature_importances_.reshape(window_size, n_features)
avg_importance = np.mean(importances, axis=0)

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': avg_importance
}).sort_values(by='Importance', ascending=False)

print("6")
# ------------------------------
# Plot top N features
# ------------------------------
top_n = 15
top_features = importance_df.head(top_n)

plt.figure(figsize=(12, 6))
plt.bar(top_features['Feature'], top_features['Importance'], color='teal')
plt.title(f"Top {top_n} Technical & Engineered Indicators (Feature Importance)")
plt.xlabel("Feature")
plt.ylabel("Average Importance over Window")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

print("7")
# ------------------------------
# Export results (optional)
# ------------------------------
top_features.to_csv('top_features.csv', index=False)
print("✅ Top features saved to: top_features.csv")

print("8")
# ------------------------------
# Print for model use
# ------------------------------
selected_features = top_features['Feature'].tolist()
print("🎯 Selected top features:", selected_features)

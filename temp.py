import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
import pandas as pd

df = pd.read_csv('C:\\Users\\junai\\Documents\\ML Projects\\Stock prediction\\Data\\indexProcessed.csv')  # Replace with your actual file path
# Convert Date to datetime
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df['EncodedIndex'] = LabelEncoder().fit_transform(df['Index'])
# Sort to maintain temporal order
df = df.sort_values(['Index', 'Date']).reset_index(drop=True)


X = df['CloseUSD'].values
y = df['CloseUSD'].values 

X_train,Y_train, X_test, Y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Define the model
model  = Sequential(
    [
        LSTM(64,return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(1, activation='linear')
        
    ]
)

model.compile(optimizer='adam', loss='mean_squared_error')

history   = model.fit(X_train,Y_train,epochs=10, batch_size=32, validation_data=(X_test, Y_test),callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
                      )

# Evaluate the model
loss = model.evaluate(X_test, Y_test)
print(f"Test Loss: {loss}")
# Plot training & validation loss
import matplotlib.pyplot as plt




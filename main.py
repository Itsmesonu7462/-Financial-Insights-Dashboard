import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import streamlit as st
import plotly.graph_objects as go

# Alpha Vantage API Key
API_KEY = '66BGIUEPCNC3I7GQ'

def fetch_stock_data_alpha_vantage(symbol):
    ts = TimeSeries(key=API_KEY, output_format='pandas')
    data, _ = ts.get_daily(symbol=symbol, outputsize='full')
    data = data[['4. close']]
    data.rename(columns={'4. close': 'Close'}, inplace=True)
    data.index = pd.to_datetime(data.index).strftime('%m/%d/%Y')
    return data.sort_index()

def fetch_latest_price(symbol):
    ts = TimeSeries(key=API_KEY, output_format='pandas')
    data, _ = ts.get_quote_endpoint(symbol=symbol)
    latest_price = float(data['05. price'][0])
    return latest_price

# Step 2: Preprocess Data
def preprocess_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))
    return scaled_data, scaler

# Step 3: Create Dataset for LSTM
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# Step 4: Build LSTM Model
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Step 5: Predict Next Value
def predict_next_value(model, scaler, data, time_step):
    # Take the last `time_step` values for prediction
    last_sequence = data[-time_step:]
    last_sequence_scaled = scaler.transform(last_sequence.reshape(-1, 1))
    input_sequence = last_sequence_scaled.reshape(1, time_step, 1)

    # Predict the next value
    predicted_scaled = model.predict(input_sequence)
    predicted_value = scaler.inverse_transform(predicted_scaled)[0, 0]
    return predicted_value

# Step 6: Plot Results
def plot_results(actual, predicted):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=actual, mode='lines', name='Actual Prices'))
    fig.add_trace(go.Scatter(y=predicted, mode='lines', name='Predicted Prices'))
    st.plotly_chart(fig)

# Streamlit Frontend
st.title("AI-Powered Financial Dashboard")
st.sidebar.title("Configuration")

# User Input
ticker = st.sidebar.text_input("Stock Ticker", value="AAPL")
time_step = st.sidebar.slider("Time Step for LSTM", min_value=10, max_value=100, value=60)

# Fetch Data
st.write(f"Fetching data for {ticker} from Alpha Vantage...")
data = fetch_stock_data_alpha_vantage(ticker)
st.write(data.tail())

# Preprocess Data
scaled_data, scaler = preprocess_data(data)
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Create Dataset
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build and Train Model
st.write("Building and training LSTM model...")
model = build_lstm_model((X_train.shape[1], 1))
model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1)

# Make Predictions
predicted_train = scaler.inverse_transform(model.predict(X_train))
predicted_test = scaler.inverse_transform(model.predict(X_test))
actual_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Fetch Latest Price and Predict Next Value
latest_price = fetch_latest_price(ticker)
st.write(f"Latest Closing Price for {ticker}: ${latest_price}")

# Append the latest price to the dataset
latest_data = np.append(scaled_data, scaler.transform([[latest_price]]))

# Predict the next price using the latest data
next_price_prediction = predict_next_value(model, scaler, latest_data, time_step)
st.write(f"Predicted Next Price: ${next_price_prediction:.2f}")

# Add date input widget to search for specific date
search_date = st.sidebar.date_input("Select Date", pd.to_datetime(data.index[-1]))

# Check if selected date is available in the data
if search_date.strftime('%m/%d/%Y') in data.index:
    date_data = data.loc[search_date.strftime('%m/%d/%Y')]
    st.write(f"Data for {search_date.strftime('%m/%d/%Y')}: {date_data['Close']}")
    # Add custom prediction for that date (if needed)
else:
    st.write(f"No data available for {search_date.strftime('%m/%d/%Y')}.")

# Plot Results
st.write("### Predictions vs Actual Data")
plot_results(actual_test.flatten(), predicted_test.flatten())

# Export Report
if st.button("Export Report"):
    with open("report.txt", "w") as f:
        f.write("AI-Powered Financial Dashboard Report\n")
        f.write(f"Ticker: {ticker}\n")
        f.write(f"Time Step: {time_step}\n")
        f.write(f"Latest Closing Price: ${latest_price}\n")
        f.write(f"Predicted Next Price: ${next_price_prediction:.2f}\n")
        f.write("Model Trained and Predictions Generated.")
    st.success("Report exported as report.txt")

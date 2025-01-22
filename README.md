

# AI-Powered Financial Dashboard

The **AI-Powered Financial Dashboard** is a web-based application built using Streamlit. It leverages LSTM (Long Short-Term Memory) models to predict stock prices using historical data fetched from Alpha Vantage. The application allows users to visualize actual vs. predicted prices, check specific historical data, and export reports summarizing predictions.

---

## Features
- **Historical Data Fetching**: Retrieve stock market data for a specified ticker using Alpha Vantage.
- **LSTM-based Prediction**: Predict future stock prices using time-series data and a trained LSTM model.
- **Search by Date**: Look up the stock’s closing price for a specific date.
- **Interactive Plots**: Visualize actual and predicted stock prices using Plotly.
- **Latest Price & Prediction**: Display the latest closing price and predict the next day’s price.
- **Report Export**: Generate a downloadable summary of predictions.

---

## Getting Started

### Prerequisites
Ensure the following are installed:
- Python 3.8+
- Alpha Vantage API Key ([Get your free API key](https://www.alphavantage.co/))
- Required Python libraries:
  ```bash
  pip install pandas numpy matplotlib streamlit plotly alpha_vantage scikit-learn tensorflow
  ```

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ai-financial-dashboard.git
   cd ai-financial-dashboard
   ```
2. Replace the `API_KEY` in the script with your Alpha Vantage API key.

---

## Usage

### Running the Application
Start the Streamlit app by running:
```bash
streamlit run app.py
```
The app will launch in your default browser at `http://localhost:8501`.

---

### How to Use
1. **Enter Stock Ticker**:
   - Input a valid stock ticker (e.g., `AAPL`) in the sidebar.
2. **Set Time Step**:
   - Adjust the time step slider (default is 60) to determine how many past data points the model considers for prediction.
3. **View Historical Data**:
   - The dashboard displays the latest fetched data for the stock.
4. **Train the LSTM Model**:
   - The app automatically builds and trains the model using the fetched data.
5. **View Predictions**:
   - Displays the latest closing price and predicts the next day’s price.
6. **Search by Date**:
   - Use the date picker in the sidebar to check the stock’s closing price on a specific date.
7. **Export Report**:
   - Click the **Export Report** button to save the key insights as a `report.txt` file.

---

## Example Workflow
1. **Input**:
   - Stock Ticker: `AAPL`
   - Time Step: `60`
2. **Process**:
   - The app fetches and processes historical stock data.
   - Trains an LSTM model with the specified time step.
3. **Output**:
   - Latest Closing Price: `$145.67`
   - Predicted Next Price: `$147.23`
   - Closing Price on 01/01/2025: `$142.85` (if available)
4. **Report**:
   ```
   AI-Powered Financial Dashboard Report
   Ticker: AAPL
   Time Step: 60
   Latest Closing Price: $145.67
   Predicted Next Price: $147.23
   Model Trained and Predictions Generated.
   ```

---

## File Structure
```
ai-financial-dashboard/
│
├── app.py                 # Main application script
├── requirements.txt       # Dependencies for the project
├── README.md              # Documentation
└── report.txt             # Exported report (generated after running)
```

---

## Troubleshooting
- **API Issues**: Ensure you’ve provided a valid Alpha Vantage API key.
- **No Data Found**: Check if the market was closed on the requested date.
- **Model Takes Long to Train**: Reduce the time step or use a smaller dataset.

---

## Future Enhancements
- Add support for multiple stock tickers simultaneously.
- Implement multi-day forecasting (e.g., predict the next 7 days).
- Include technical indicators like moving averages or RSI.
- Add functionality for real-time data fetching.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.


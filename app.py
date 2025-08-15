import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from flask import Flask, request, render_template
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import load_model

# Load & Preprocess Dataset
df = pd.read_csv("FinalDataset1.csv", encoding='utf-8')
df.columns = df.columns.str.strip().str.replace(r'\s+', '_', regex=True)

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date']).sort_values(by='Date').set_index('Date')
df['Stock'] = df['Stock'].astype(str).str.strip().str.upper()

numeric_cols = ['Open', 'High', 'Low', 'Close']
df[numeric_cols] = df[numeric_cols].replace({',': ''}, regex=True).astype(float)

scaler = MinMaxScaler(feature_range=(0, 1))
app = Flask(__name__)

# Load Trained Models
lstm_model = load_model("lstm_model.h5")
rnn_model = load_model("rnn_model.h5")

@app.route('/')
def home():
    return render_template('index.html', stocks=df['Stock'].unique())

@app.route('/predict', methods=['POST'])
def predict():
    stock_name = request.form['stock_name'].strip().upper()
    stock_data = df[df['Stock'] == stock_name]
    
    if stock_data.empty:
        return f"Stock '{stock_name}' not found in dataset!"

    # Preprocess Data
    scaled_data = scaler.fit_transform(stock_data['Close'].values.reshape(-1, 1))
    if len(scaled_data) < 60:
        return "Not enough data to make predictions."

    X_lstm = np.array([scaled_data[i-60:i] for i in range(60, len(scaled_data))])
    X_rnn = np.array([scaled_data[i-60:i] for i in range(60, len(scaled_data))])

    # Predictions using Deep Learning Models
    y_pred_lstm = lstm_model.predict(X_lstm, batch_size=32).flatten()
    y_pred_rnn = rnn_model.predict(X_rnn, batch_size=32).flatten()

    # Train & Predict using SVM
    svm_model = SVR(kernel='rbf', C=1000, gamma=0.1)
    X_svm = np.arange(len(scaled_data)).reshape(-1, 1)
    svm_model.fit(X_svm[:-60], scaled_data[:-60].flatten())  # Train on historical data
    y_pred_svm = svm_model.predict(X_svm[60:])

    # Inverse Scaling
    y_pred_lstm = scaler.inverse_transform(y_pred_lstm.reshape(-1, 1)).flatten()
    y_pred_rnn = scaler.inverse_transform(y_pred_rnn.reshape(-1, 1)).flatten()
    y_pred_svm = scaler.inverse_transform(y_pred_svm.reshape(-1, 1)).flatten()

    # Align actual prices with predictions
    actual_prices = stock_data['Close'].values[60:len(y_pred_lstm) + 60]

    # Calculate Accuracy Metrics
    def calculate_metrics(actual, predicted):
        return {
            "MAE": mean_absolute_error(actual, predicted),
            "MSE": mean_squared_error(actual, predicted),
            "RMSE": np.sqrt(mean_squared_error(actual, predicted))
        }

    lstm_metrics = calculate_metrics(actual_prices, y_pred_lstm)
    rnn_metrics = calculate_metrics(actual_prices, y_pred_rnn)
    svm_metrics = calculate_metrics(actual_prices, y_pred_svm)
    
    errors = {
        "LSTM": lstm_metrics["MAE"],
        "RNN": rnn_metrics["MAE"],
        "SVM": svm_metrics["MAE"]
    }
    best_model = min(errors, key=errors.get)

    # Save Graphs
    def create_graph(predictions, title, color, metrics):
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=stock_data.index[60:len(predictions) + 60],
            open=stock_data['Open'][60:len(predictions) + 60],
            high=stock_data['High'][60:len(predictions) + 60],
            low=stock_data['Low'][60:len(predictions) + 60],
            close=stock_data['Close'][60:len(predictions) + 60],
            name='Actual Prices'
        ))
        fig.add_trace(go.Scatter(
            x=stock_data.index[60:len(predictions) + 60],
            y=predictions, mode='lines',
            name=f"{title} (MAE: {metrics['MAE']:.2f}, RMSE: {metrics['RMSE']:.2f})",
            line=dict(color=color)
        ))
        fig.update_layout(
            title=f"{title} Predictions",
            xaxis_title='Date',
            yaxis_title='Price',
            width=1200,    # Increase graph width
            height=800     # Increase graph height
        )
        fig.write_html(f"static/{title}_graph.html")

    create_graph(y_pred_svm, "SVM", "blue", svm_metrics)
    create_graph(y_pred_lstm, "LSTM", "red", lstm_metrics)
    create_graph(y_pred_rnn, "RNN", "green", rnn_metrics)

    return render_template('result.html', stock_name=stock_name, best_model=best_model, 
                           svm_graph="static/SVM_graph.html", 
                           lstm_graph="static/LSTM_graph.html", 
                           rnn_graph="static/RNN_graph.html")

@app.route('/future_predict', methods=['POST'])
def future_predict():
    stock_name = request.form['stock_name'].strip().upper()
    n_days = int(request.form['n_days'])
    stock_data = df[df['Stock'] == stock_name]
    
    if stock_data.empty:
        return f"Stock '{stock_name}' not found!"
    
    scaled_data = scaler.fit_transform(stock_data['Close'].values.reshape(-1, 1))
    X_input = scaled_data[-60:].reshape(1, 60, 1)
    
    future_predictions = []
    future_dates = pd.date_range(start=stock_data.index[-1] + pd.Timedelta(days=1), periods=n_days)
    
    for _ in range(n_days):
        pred = lstm_model.predict(X_input)[0][0]
        future_predictions.append(pred)
        X_input = np.append(X_input[:, 1:, :], [[[pred]]], axis=1)
    
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=future_dates, y=future_predictions.flatten(), mode='lines+markers', name='Future Predictions'))
    fig.update_layout(title="Future Stock Price Prediction", xaxis_title="Date", yaxis_title="Predicted Price")
    fig.write_html("static/Future_graph.html")
    
    return render_template('future_result.html', future_graph="static/Future_graph.html")

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contactus.html')

if __name__ == "__main__":
    app.run(debug=True)

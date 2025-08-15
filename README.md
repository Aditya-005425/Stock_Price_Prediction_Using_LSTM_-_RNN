# 📈 Stock Market Prediction using LSTM & RNN

## 📌 Overview
This project predicts stock prices using deep learning techniques, specifically **Long Short-Term Memory (LSTM)** and **Recurrent Neural Networks (RNN)**.  
It processes historical stock data, trains models, and generates visual predictions for future price trends.  
Built using **Python**, **TensorFlow/Keras**, and **Pandas**.

---

## 🛠️ Project Structure
project/
│
├── data/
│ └── historical_stock_data.csv # Historical stock price dataset
│
├── notebooks/
│ ├── data_preprocessing.ipynb # Data cleaning and preparation
│ ├── lstm_model.ipynb # LSTM model training
│ ├── rnn_model.ipynb # RNN model training
│ └── evaluation.ipynb # Model evaluation and comparison
│
├── src/
│ ├── preprocess.py # Data preprocessing script
│ ├── train_lstm.py # Train LSTM model
│ ├── train_rnn.py # Train RNN model
│ ├── predict.py # Predict future stock prices
│ └── visualize.py # Visualization script
│
├── results/
│ ├── lstm_predictions.png # LSTM prediction chart
│ └── rnn_predictions.png # RNN prediction chart
│
├── requirements.txt # Python dependencies
└── README.md # Project documentation

---

## ⚙️ Setup Instructions

### 1️⃣ Install Dependencies
Make sure **Python 3.8+** is installed, then run:
```bash
pip install -r requirements.txt

2️⃣ Prepare Dataset
Download historical stock data from:
Yahoo Finance
Kaggle Datasets
Save it as historical_stock_data.csv in the data/ folder.

3️⃣ Run Preprocessing
python src/preprocess.py

4️⃣ Train Models
Train LSTM:
python src/train_lstm.py
Train RNN:
python src/train_rnn.py

5️⃣ Make Predictions
python src/predict.py

6️⃣ Visualize Results
python src/visualize.py


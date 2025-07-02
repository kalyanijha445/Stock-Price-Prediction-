from flask import Flask, render_template, request
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# Define same model architecture
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=2, batch_first=True)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

# Load model and scaler
model = LSTMModel()
model.load_state_dict(torch.load("lstm_model.pt"))
model.eval()

scaler = joblib.load("scaler.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["GET"])
def predict_page():
    return render_template("predict.html")

@app.route("/result", methods=["POST"])
def result():
    try:
        # Load data
        df = pd.read_csv("AAPL.csv")
        close_prices = df["Close"].values[-60:].reshape(-1, 1)

        # Sanity check
        if len(close_prices) < 60:
            return "Not enough data to predict. Need at least 60 days of prices.", 400

        # Scale and reshape
        scaled_seq = scaler.transform(close_prices)
        sequence = torch.tensor(scaled_seq, dtype=torch.float32).unsqueeze(0)  # (1, 60, 1)

        # Predict
        with torch.no_grad():
            predicted_scaled = model(sequence).item()

        predicted_price = scaler.inverse_transform([[predicted_scaled]])[0][0]

        return render_template("result.html", predicted_price=round(predicted_price, 2))

    except Exception as e:
        return f"Error: {str(e)}", 500

if __name__ == "__main__":
    app.run(debug=True)

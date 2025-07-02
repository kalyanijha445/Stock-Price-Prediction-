import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import joblib

# Load stock data
df = pd.read_csv("AAPL.csv")
data = df["Close"].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)
joblib.dump(scaler, "scaler.pkl")  # Save the scaler

# Function to create sequences
def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])  # Flatten each sequence
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# Create dataset
X, y = create_sequences(scaled_data)

# Sanity check
if X.shape[0] == 0:
    raise ValueError("Not enough data! Please use a CSV with at least 70 rows.")

# Convert to PyTorch tensors and reshape
X = torch.from_numpy(X).float().unsqueeze(-1)  # Shape: (samples, seq_len, 1)
y = torch.from_numpy(y).float().unsqueeze(-1)  # Shape: (samples, 1)

# LSTM Model Definition
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=2, batch_first=True)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Last time step
        return self.fc(out)

# Initialize model
model = LSTMModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 10
for epoch in range(epochs):
    model.train()
    output = model(X)
    loss = criterion(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Save model
torch.save(model.state_dict(), "lstm_model.pt")
print("✅ Model and scaler saved successfully.")

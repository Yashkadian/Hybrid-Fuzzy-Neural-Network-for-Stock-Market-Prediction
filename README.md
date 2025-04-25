# Hybrid-Fuzzy-Neural-Network-for-Stock-Market-Prediction
Hybrid Fuzzy-Neural Network for Stock Market Prediction
📌 Description
This project implements a Hybrid Fuzzy-Neural Network model to predict stock market trends by combining fuzzy logic for uncertainty handling and neural networks for pattern recognition. The system consists of:

Backend (backend.py): A Python-based API that processes stock data and runs predictions.

Frontend: A web interface to input stock symbols and visualize predictions.

🚀 Quick Start Guide
Prerequisites
✔ Python 3.8+ (Install from python.org)
✔ VS Code (or any editor with Live Server support)
✔ Required Python libraries 

Step 1: Start the Backend API
Open a terminal in the project folder.

Run the backend server:

bash
python backend.py
Keep this terminal open—the API must stay running!

Step 2: Launch the Frontend
Open the project in VS Code.

Right-click on index.html → Open with Live Server (or click Go Live in the status bar).

Step 3: Use the Predictor
Enter a stock symbol (e.g., AAPL, TSLA).

Click Predict to see:

Historical trends

Fuzzy-neural network predictions

Confidence intervals

🛠 Technical Details
Model Architecture
Fuzzy Logic Layer: Handles volatility and market uncertainty.

LSTM Neural Network: Learns temporal patterns in stock data.

Hybrid Training: Combines fuzzy rules with backpropagation.

Data Sources
Yahoo Finance API (yfinance)

Custom datasets for fuzzy rule calibration

📊 Expected Output
After running a prediction, you’ll see:
✅ Predicted Price (Next 5 days)
✅ Buy/Sell/Hold Recommendation
✅ Volatility Analysis (Fuzzy logic output)

# attack_detector.py
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
from typing import List, Dict, Any
import logging

# Load paths
MODEL_PATH = os.path.join(os.getcwd(), 'models', 'lstm_model.pt')
SCALER_PATH = os.path.join(os.getcwd(), 'models', 'scaler.pkl')
FEATURES_PATH = os.path.join(os.getcwd(), 'models', 'features.pkl')

# Define the model class
class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, 64, batch_first=True)
        self.fc = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return self.sigmoid(x)

# Load model and preprocessing tools
try:
    model = LSTMModel(input_size=28)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    scaler = joblib.load(SCALER_PATH)
    selected_features = joblib.load(FEATURES_PATH)
except Exception as e:
    logging.error(f"Error loading model or preprocessing tools: {str(e)}")
    raise

# Preprocess and predict
def detect_attack(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        if not data:
            return {"status": "error", "message": "Empty data"}

        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Validate required features
        missing_features = [f for f in selected_features if f not in df.columns]
        if missing_features:
            return {"status": "error", "message": f"Missing required features: {missing_features}"}

        # Select and scale features
        df = df[selected_features]
        scaled = scaler.transform(df)
        
        # Process in batches for better performance
        batch_size = 32
        predictions = []
        
        for i in range(0, len(scaled), batch_size):
            batch = scaled[i:i + batch_size]
            reshaped = batch.reshape((batch.shape[0], 1, batch.shape[1]))
            tensor_data = torch.tensor(reshaped, dtype=torch.float32)
            
            if torch.cuda.is_available():
                tensor_data = tensor_data.cuda()
            
            with torch.no_grad():
                outputs = model(tensor_data)
                batch_predictions = (outputs > 0.5).float()
                predictions.extend(batch_predictions.cpu().numpy())

        attack_detections = []
        for i, pred in enumerate(predictions):
            if pred == 1.0:
                row = df.iloc[i]
                attack_detections.append({
                    "src_bytes": row.get('src_bytes', 'N/A'),
                    "dst_bytes": row.get('dst_bytes', 'N/A'),
                    "count": row.get('count', 'N/A'),
                    "message": "Attack detected"
                })

        if attack_detections:
            return {"status": "ok", "attack_detected": attack_detections}
        else:
            return {"status": "ok", "message": "No attack detected"}

    except Exception as e:
        logging.error(f"Error in attack detection: {str(e)}")
        return {"status": "error", "message": f"Error processing data: {str(e)}"}

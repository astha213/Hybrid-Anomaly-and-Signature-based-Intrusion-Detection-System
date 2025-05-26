# Hybrid-Anomaly-and-Signature-based-Intrusion-Detection-System using Isolation Forest and LSTM
This project implements a Hybrid Network Intrusion Detection System that combines anomaly detection and deep learning to identify and mitigate network attacks, including real-time Denial of Service (DoS) attack simulation and prevention.

# 🔍 Project Overview

- Combines Isolation Forest (unsupervised anomaly detection) with LSTM (deep learning for sequential data).
- Performs feature selection using Recursive Feature Elimination (RFE) and Mutual Information (MI) to improve detection accuracy.
- Supports real-time detection and automatic IP blocking during simulated DoS attacks.
- Uses the NSL-KDD dataset for training and evaluation, with support for other datasets like UNSW-NB15 and CICIDS2017.

# ⚙️ Features

- Detects both known and unknown attack patterns.
- Filters redundant features to enhance model performance.
- Hybrid model architecture:
  - Isolation Forest detects outliers.
  - LSTM learns temporal traffic patterns.
- Real-time DoS attack simulation and blocking via Python scripts.

# To simulate a DoS attack and test real-time blocking:
python app.py

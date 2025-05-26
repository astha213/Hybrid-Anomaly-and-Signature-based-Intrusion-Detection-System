# Hybrid-Anomaly-and-Signature-based-Intrusion-Detection-System using Isolation Forest and LSTM
This project implements a Hybrid Network Intrusion Detection System that combines anomaly detection and deep learning to identify and mitigate network attacks, including real-time Denial of Service (DoS) attack simulation and prevention.

# 🔍 Project Overview

- Combines Isolation Forest (unsupervised anomaly detection) with LSTM (deep learning for sequential data).
- Performs feature selection using Recursive Feature Elimination (RFE) and Mutual Information (MI) to improve detection accuracy.
- Supports real-time detection and automatic IP blocking during simulated DoS attacks.
- Uses the NSL-KDD dataset for training and evaluation, with support for other datasets like UNSW-NB15 and CICIDS2017.
- Applies 10-fold cross-validation on NSL-KDD’s separate train and test sets to ensure unbiased model evaluation and better generalization.
  
# ⚙️ Features

- Detects both known and unknown attack patterns.
- Filters redundant features to enhance model performance.
- Hybrid model architecture:
  - Isolation Forest detects outliers.
  - LSTM learns temporal traffic patterns.
- Real-time DoS attack simulation and blocking via Python scripts.

# To simulate a DoS attack and test real-time blocking:
python app.py


# Selected Features (RFE+MI)
![Screenshot 2025-04-24 102852](https://github.com/user-attachments/assets/0d0695ba-d8f8-47d7-b30f-c8e1caabd9f9)


# Proposed Model Graph
![image](https://github.com/user-attachments/assets/9b418de8-4fe3-4221-a6c3-f4abbc5d96f0)


# Real-time Denial of Service (DoS) attack simulation and IP Blocking
https://github.com/user-attachments/assets/21106cdb-4632-487c-92dd-42ffbb1f75d2


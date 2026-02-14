# Credit Card Fraud Detection with Isolation Forest

## Overview
This project implements an unsupervised fraud detection model using the Kaggle Credit Card Fraud dataset (2013). The model uses Isolation Forest to detect anomalous transactions.

## Problem
Extreme class imbalance (0.17% fraud) makes traditional supervised models unreliable.

## Method
- Isolation Forest (contamination=0.003)
- Threshold optimization (99.3 percentile)
- Evaluation using Recall, Precision, F1-score, ROC-AUC

## Results
- Recall: 51%
- Precision: 12.5%
- F1-score: 0.20
- ROC-AUC: 0.96

## Technologies Used
- Python
- Scikit-learn
- Pandas
- Seaborn
- PCA
- Matplotlib

## Key Takeaway
The model detects over 50% of fraud cases without labeled training data.

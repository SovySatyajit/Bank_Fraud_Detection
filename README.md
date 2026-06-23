# Graph-Based Banking Fraud Detection using Graph Neural Networks

## Overview

This project implements a graph-based banking fraud detection system that models financial transactions as a network of interconnected accounts and uses Graph Neural Networks (GNNs) to identify suspicious activity.

Unlike traditional machine learning approaches that analyze transactions independently, this system captures relationships between accounts, enabling the detection of hidden fraud patterns, suspicious transaction chains, and connected fraudulent entities.

---

## Features

* Transaction graph construction from banking transaction data
* Graph-based feature engineering
* Graph Neural Network (GraphSAGE) for learning account representations
* Fraud risk prediction using learned graph embeddings
* Fraud probability scoring
* Visualization of transaction networks
* Modular and scalable architecture

---

## Technology Stack

### Data Processing

* Python
* Pandas
* NumPy

### Graph Processing

* NetworkX
* PyTorch Geometric

### Machine Learning

* Scikit-Learn
* XGBoost

### Deep Learning

* PyTorch
* GraphSAGE

### Visualization

* Matplotlib
* Streamlit

---

## Project Architecture

Transaction Dataset
→ Data Preprocessing
→ Transaction Graph Construction
→ Graph Feature Engineering
→ GraphSAGE Embedding Generation
→ Fraud Classification
→ Fraud Risk Scoring
→ Visualization Dashboard

---

## Dataset

This project uses the PaySim synthetic financial transaction dataset.

Dataset includes:

* Sender Account
* Receiver Account
* Transaction Amount
* Transaction Type
* Account Balances
* Fraud Labels

Note: The dataset is not included in this repository due to size limitations.

---

## Project Structure (NOT EVERYTHING IS DONE)

bank_fraud_detection/

├── data/

├── src/

│ ├── graph_builder.py

│ ├── feature_engineering.py

│ ├── graphsage_train.py

│ ├── fraud_classifier.py

│ └── predict.py

├── outputs/

├── app/

│ └── dashboard.py

├── requirements.txt

├── README.md

└── .gitignore

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd bank_fraud_detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Current Progress

* [x] Dataset loading
* [x] Transaction graph construction
* [ ] Graph feature engineering
* [ ] GraphSAGE training
* [ ] Fraud classification
* [ ] Dashboard development

---

## Future Improvements

* Real-time transaction streaming
* Fraud ring detection
* Heterogeneous graph modeling
* Explainable AI for fraud decisions
* Advanced graph anomaly detection
* Production deployment using Docker

---

## Objective

Build a practical graph-based fraud detection system that demonstrates how Graph Neural Networks can leverage transaction relationships to improve fraud detection beyond traditional machine learning techniques.

---

## Author

Sovyam Satyajit

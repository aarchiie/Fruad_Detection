# Fraud Detection System

A hybrid fraud detection tool using supervised (Random Forest, MLP), unsupervised (Isolation Forest), graph-based, time-series, and rule-based methods.

## Prerequisites
- Python 3.7+
- Packages: `pandas`, `numpy`, `networkx`, `scikit-learn`, `imblearn`
- Dataset: `update_data.csv`

## Installation
1. Install Python from [python.org](https://www.python.org/downloads/).
2. Run: `pip install pandas numpy networkx scikit-learn imblearn`
3. Place `main.py` and `update_data.csv` in the same directory.

## Dataset
Requires `update_data.csv` with columns: `customer_id`, `dest_id`, `amount`, `type`, `kyc_verified`, `customer_risk_score`, `account_age_days`, `is_pep`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`, `timestamp`, `isFraud`.

## Running
1. Navigate to directory: `cd path/to/directory`
2. Execute: `python main.py`
3. Input transaction details when prompted.

## Usage
- View model accuracy and classification reports.
- Enter: transaction type, customer ID, destination ID, amount.
- Get fraud prediction, score, and risk factors.

## Features
- Supervised and unsupervised learning
- Graph and time-series analysis
- Business/KYC rules
- Hybrid prediction (threshold: 0.6)

## Troubleshooting
- Ensure `update_data.csv` exists.
- Verify package installation.
- Match input IDs to dataset.

## Limitations
- Needs pre-existing dataset.
- Single-run (edit `main()` for multiple transactions).

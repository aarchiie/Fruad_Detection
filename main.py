import pandas as pd
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from collections import Counter
import warnings

warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv("update_data.csv")
df['timestamp'] = '2025-01-01 ' + df['timestamp']
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', format='%Y-%m-%d %H:%M:%S')

df_for_lookup = df[
    ['customer_id', 'dest_id', 'amount', 'kyc_verified', 'customer_risk_score', 'account_age_days', 'is_pep',
     'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'timestamp']]

df.sort_values(by='timestamp', inplace=True)
df.drop(columns=['timestamp'], inplace=True)

# Encode categorical variables
label_encoders = {}
categorical_cols = ['type', 'kyc_verified', 'is_pep']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

features_to_remove = ['oldbalanceOrg', 'amount', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'customer_id',
                      'dest_id']
X = df.drop(columns=['isFraud'] + features_to_remove)
y = df['isFraud']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# SMOTE for handling class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42,
                                                    stratify=y_resampled)

# Models
supervised_model = RandomForestClassifier(n_estimators=100, random_state=42)
supervised_model.fit(X_train, y_train)

unsupervised_model = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
unsupervised_model.fit(X_train)

mlp_model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=200, random_state=42)
mlp_model.fit(X_train, y_train)


# Model Evaluation
def evaluate_models():
    # Random Forest Evaluation
    rf_pred = supervised_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    print("\nRandom Forest Model Evaluation:")
    print(f"Accuracy: {rf_accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, rf_pred))

    # MLP Evaluation
    mlp_pred = mlp_model.predict(X_test)
    mlp_accuracy = accuracy_score(y_test, mlp_pred)
    print("\nMLP Model Evaluation:")
    print(f"Accuracy: {mlp_accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, mlp_pred))

    # Isolation Forest Evaluation (converting -1/1 to 1/0 for consistency)
    iso_pred = (unsupervised_model.predict(X_test) == -1).astype(int)
    iso_accuracy = accuracy_score(y_test, iso_pred)
    print("\nIsolation Forest Model Evaluation:")
    print(f"Accuracy: {iso_accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, iso_pred))


# Graph-Based Fraud Detection
def build_transaction_graph(df):
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(row['customer_id'], row['dest_id'], weight=row['amount'])
    return G


graph = build_transaction_graph(df_for_lookup)


# Time Series Anomalies
def detect_time_series_anomalies(df, window=5):
    df['rolling_mean'] = df['amount'].rolling(window=window).mean()
    df['rolling_std'] = df['amount'].rolling(window=window).std()
    df['z_score'] = (df['amount'] - df['rolling_mean']) / df['rolling_std']
    df['time_anomaly'] = (df['z_score'].abs() > 3).astype(int)
    return df


df_for_lookup = detect_time_series_anomalies(df_for_lookup)


# Business Rules
def apply_business_rules(transaction):
    rules_triggered = []
    if transaction['amount'] > 10000 and transaction['oldbalanceOrg'] == 0:
        rules_triggered.append("Large transfer from empty account")
    if transaction['oldbalanceOrg'] - transaction['amount'] != transaction['newbalanceOrig']:
        rules_triggered.append("Balance Mismatch")
    return rules_triggered


# KYC and AML Rules
def apply_kyc_rules(transaction):
    kyc_issues = []
    if transaction['kyc_verified'] == 0:
        kyc_issues.append("KYC not verified")
    if transaction['is_pep'] == 1 and transaction['amount'] > 10000:
        kyc_issues.append("High-value transaction by PEP")
    if transaction['customer_risk_score'] > 0.8:
        kyc_issues.append("High-risk customer")
    if transaction['account_age_days'] < 30 and transaction['amount'] > 5000:
        kyc_issues.append("New Account Monitoring")
    return kyc_issues


# Risk Factors
def get_top_risk_factors(transaction, hybrid_prediction, supervised_prediction, unsupervised_prediction):
    risk_factors = []
    if hybrid_prediction == 1 and supervised_prediction == 1 and unsupervised_prediction == 1:
        risk_factors.append("High Fraud Confidence ( >70%)")
    if transaction['amount'] > 0.9 * transaction['oldbalanceOrg']:
        risk_factors.append("Balance Drainage")
    return risk_factors


# Hybrid Model Prediction
def hybrid_fraud_detection(transaction):
    input_data = pd.DataFrame([transaction])[X.columns]
    input_scaled = scaler.transform(input_data)
    supervised_pred = supervised_model.predict(input_scaled)[0]
    mlp_pred = mlp_model.predict(input_scaled)[0]
    unsupervised_pred = (unsupervised_model.predict(input_scaled)[0] == -1).astype(int)
    hybrid_pred = np.logical_or(supervised_pred, unsupervised_pred).astype(int)
    time_anomaly = transaction['time_anomaly']
    graph_risk = transaction['customer_id'] in graph and transaction['dest_id'] in graph and graph.has_edge(
        transaction['customer_id'], transaction['dest_id'])
    return hybrid_pred or time_anomaly or graph_risk


# Get User Transaction
def get_user_transaction():
    transaction = {}
    transaction['type'] = \
        label_encoders['type'].transform([input("Enter transaction type (e.g., TRANSFER, CASH_OUT): ")])[0]
    transaction['customer_id'] = input("Enter customer ID: ")
    transaction['dest_id'] = input("Enter destination ID: ")
    transaction['amount'] = float(input("Enter transaction amount: "))

    customer_data = df_for_lookup[(df_for_lookup['customer_id'] == transaction['customer_id']) & (
            df_for_lookup['dest_id'] == transaction['dest_id'])]
    if customer_data.empty:
        print("Transaction details not found. Cannot process fraud detection.")
        return

    transaction.update(customer_data.iloc[0].to_dict())

    print("\nBusiness Rules Applied:", apply_business_rules(transaction))
    print("KYC Rules Applied:", apply_kyc_rules(transaction))

    input_data = pd.DataFrame([transaction])[X.columns]
    input_scaled = scaler.transform(input_data)

    supervised_pred = supervised_model.predict(input_scaled)[0]
    mlp_pred = mlp_model.predict(input_scaled)[0]
    unsupervised_pred = (unsupervised_model.predict(input_scaled)[0] == -1).astype(int)
    hybrid_prediction = np.logical_or(supervised_pred, unsupervised_pred).astype(int)

    graph_risk = transaction['customer_id'] in graph and transaction['dest_id'] in graph and graph.has_edge(
        transaction['customer_id'], transaction['dest_id'])
    time_anomaly = transaction.get('time_anomaly', 0)

    fraud_score = (0.35 * supervised_pred) + (0.25 * mlp_pred) + (0.2 * unsupervised_pred) + (0.1 * graph_risk) + (
            0.1 * time_anomaly)
    final_fraud_prediction = fraud_score > 0.6

    print("\nPredictions:")
    print("Supervised Prediction:", 'Fraud' if supervised_pred else 'Not Fraud')
    print("MLP Prediction:", 'Fraud' if mlp_pred else 'Not Fraud')
    print("Unsupervised Prediction:", 'Fraud' if unsupervised_pred else 'Not Fraud')
    print("Graph-Based Detection:", 'Suspicious' if graph_risk else 'Normal')
    print("Time-Series Anomaly Detection:", 'Anomalous' if time_anomaly else 'Normal')
    print("Final Fraud Prediction:", 'Fraud' if final_fraud_prediction else 'Not Fraud')
    print(f"Fraud Score: {fraud_score:.4f}")
    print("Top Risk Factors:", get_top_risk_factors(transaction, hybrid_prediction, supervised_pred, unsupervised_pred))


def main():
    print("Evaluating Models...")
    evaluate_models()
    print("\nStarting Transaction Fraud Detection...")
    get_user_transaction()


if __name__ == "__main__":
    main()
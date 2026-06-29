import pandas as pd
import numpy as np
import os
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, average_precision_score, confusion_matrix

BASE_PATH = r"D:\Masters\Thesis\Block 3\Thesis work"
PROCESSED_PATH = os.path.join(BASE_PATH, "Data", "Processed", "dataset1")
AUG_PATH = os.path.join(BASE_PATH, "Data", "augmented")
TEST_PATH =os.path.join(PROCESSED_PATH, "D1_test.csv")
REPORTS_PATH = os.path.join(BASE_PATH, "Reports")
os.makedirs(REPORTS_PATH, exist_ok=True)
candidate_files = {
    "Baseline": {"path": os.path.join(PROCESSED_PATH, "D1_Train.csv"), "params": {"learning_rate":0.1, "subsample": 1.0}},
    "SMOTETomek": {"path": os.path.join(AUG_PATH, "D1_Train_SMOTETomek.csv"), "params": {"learning_rate":0.1, "subsample": 0.8}},
    "CTGAN_XGBoost": {"path": os.path.join(AUG_PATH, "D1_Train_CTGAN_700.csv"), "params": {"learning_rate":0.1, "subsample": 1.0}},
    "TVAEGAN_XGBoost": {"path": os.path.join(AUG_PATH, "D1_Train_TVAEGAN300.csv"), "params": {"learning_rate":0.1, "subsample": 0.8}}}

def run_test(test_csv_path, output_filename="D1_Test.csv", target_col="status"):
    df_test = pd.read_csv(test_csv_path)
    X_test = df_test.drop(columns=[target_col]).values
    y_test = df_test[target_col].values
    results = []
    for name, config in candidate_files.items():
        df_train = pd.read_csv(os.path.join(AUG_PATH, config['path']))
        X_train = df_train.drop(columns=[target_col]).values
        y_train = df_train[target_col].values
        model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=config['params']['learning_rate'], subsample=config['params']['subsample'], random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        results.append({
            "Method_Model": name,
            "Accuracy": round(accuracy_score(y_test, preds), 4),
            "F1-Score": round(f1_score(y_test, preds), 4),
            "PR-AUC": round(average_precision_score(y_test, probs), 4),
            "Precision": round(precision_score(y_test, preds), 4),
            "Recall": round(recall_score(y_test, preds), 4),
            "Confusion Matrix": str(confusion_matrix(y_test, preds).tolist())})
    pd.DataFrame(results).to_csv(os.path.join(REPORTS_PATH, "D1_Test.csv"), index=False)
if __name__ == "__main__":
    run_test(TEST_PATH, "D1_Test.csv")

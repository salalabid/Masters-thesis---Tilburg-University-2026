import pandas as pd
import numpy as np
import os
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, average_precision_score, confusion_matrix

BASE_PATH = r"D:\Masters\Thesis\Block 3\Thesis work"
PROCESSED_PATH = os.path.join(BASE_PATH, "Data", "processed", "dataset2")
AUG_PATH = os.path.join(BASE_PATH, "Data", "augmented")
TEST_PATH = os.path.join(PROCESSED_PATH, "D2_Test.csv")
REPORTS_PATH = os.path.join(BASE_PATH, "Reports")
os.makedirs(REPORTS_PATH, exist_ok=True)
candidate_files = {
    "XGBoost_Baseline": {"train_dir": PROCESSED_PATH, "file": "D2_Train.csv"},
    "CTGAN_XGBoost": {"train_dir": AUG_PATH, "file": "D2_Train_CTGAN.csv"},
    "TVAEGAN_XGBoost": {"train_dir": AUG_PATH, "file": "D2_Train_TVAEGAN.csv"}}

def run_test(target_col):
    df_test = pd.read_csv(TEST_PATH)
    X_test = df_test.drop(columns=[target_col]).values
    y_test = df_test[target_col].values
    results = []
    for name, config in candidate_files.items():
        train_file_path = os.path.join(config['train_dir'], config['file'])
        df_train = pd.read_csv(train_file_path)
        X_train = df_train.drop(columns=[target_col]).values
        y_train = df_train[target_col].values
        model = XGBClassifier(
            n_estimators=100, 
            max_depth=4, 
            learning_rate=0.1, 
            subsample=1.0, 
            random_state=42)
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
    report_df = pd.DataFrame(results)
    report_df.to_csv(os.path.join(REPORTS_PATH, "D2_Test.csv"), index=False)
    print(report_df[['Method_Model', 'Accuracy', 'F1-Score', 'PR-AUC', 'Precision', 'Recall']])
if __name__ == "__main__":
    run_test("status")
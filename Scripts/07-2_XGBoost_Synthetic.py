import pandas as pd
import numpy as np
import os
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, average_precision_score, confusion_matrix

BASE_PATH = r"D:\Masters\Thesis\Block 3\Thesis work"
AUG_PATH = os.path.join(BASE_PATH, "Data", "augmented")
VAL_PATH = os.path.join(BASE_PATH, "Data", "processed", "dataset1", "D1_Val.csv")
REPORTS_PATH = os.path.join(BASE_PATH, "Reports")
synth_files = {"CTGAN": "D1_Synthetic_CTGAN.csv", "TVAEGAN": "D1_Synthetic_TVAEGAN.csv"}

def XGBoost_Synthetic(target_col):
    df_val = pd.read_csv(VAL_PATH)
    X_val = df_val.drop(columns=[target_col]).values
    y_val = df_val[target_col].values

    results = []

    for label, filename in synth_files.items():
        file_path = os.path.join(AUG_PATH, filename)
        df_train = pd.read_csv(file_path)
        X_train = df_train.drop(columns=[target_col]).values
        y_train = df_train[target_col].values

        model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, subsample=1.0, random_state=42)
        model.fit(X_train, y_train)

        probs = model.predict_proba(X_val)[:, 1]
        preds = model.predict(X_val)

        results.append({
            "Method": label,
            "Accuracy": round(accuracy_score(y_val, preds), 4),
            "F1-Score": round(f1_score(y_val, preds), 4),
            "PR-AUC": round(average_precision_score(y_val, probs), 4),
            "Precision":round(precision_score(y_val, preds), 4),
            "Recall":round(recall_score(y_val, preds), 4),
            "Confusion Matrix":str(confusion_matrix(y_val, preds).tolist())})
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(REPORTS_PATH, "D1_XGB_Synthetic.csv"), index=False)

if __name__ == "__main__":
    XGBoost_Synthetic("status")
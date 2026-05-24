import pandas as pd
import numpy as np
import os
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, average_precision_score, confusion_matrix

def baseline(target_col):
    df_train = pd.read_csv(r"D:\Masters\Thesis\Block 3\Thesis work\Data\processed\dataset1\D1_Train.csv")
    df_val = pd.read_csv(r"D:\Masters\Thesis\Block 3\Thesis work\Data\processed\dataset1\D1_Val.csv")
    
    X_train = df_train.drop(columns=[target_col]).values
    y_train = df_train[target_col].values
    X_val = df_val.drop(columns=[target_col]).values
    y_val = df_val[target_col].values

    baseline_results = []

    svm = SVC(probability=True, random_state=42)
    svm.fit(X_train, y_train)
    svm_probs = svm.predict_proba(X_val)[:, 1]
    svm_preds = svm.predict(X_val)

    xgb = XGBClassifier(random_state=42)
    xgb.fit(X_train, y_train)
    xgb_probs = xgb.predict_proba(X_val)[:, 1]
    xgb_preds = xgb.predict(X_val)

    tn = TabNetClassifier(verbose=0)
    tn.fit(X_train=X_train, y_train=y_train, eval_set=[(X_val, y_val)], 
           max_epochs=100, patience=15)
    tn_probs = tn.predict_proba(X_val)[:, 1]
    tn_preds = tn.predict(X_val)

    models = [("SVM", svm_probs, svm_preds), ("XGBoost", xgb_probs, xgb_preds), ("TabNet", tn_probs, tn_preds)]
    
    for name, probs, preds in models:
        baseline_results.append({
            "Model": name,
            "Accuracy": round(accuracy_score(y_val, preds), 4),
            "F1-Score": round(f1_score(y_val, preds), 4),
            "PR-AUC": round(average_precision_score(y_val, probs), 4),
            "Precision": round(precision_score(y_val, preds), 4),
            "Recall": round(recall_score(y_val, preds), 4),
            "Confusion Matrix": str(confusion_matrix(y_val, preds).tolist())
        })

    results_df = pd.DataFrame(baseline_results)
    export_file = r"D:\Masters\Thesis\Block 3\Thesis work\Reports\D1_baseline.csv"
    results_df.to_csv(export_file, index=False)

    print(results_df[["Model", "PR-AUC", "F1-Score", "Recall"]])

if __name__ == "__main__":
    baseline("status")
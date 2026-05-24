import pandas as pd
import numpy as np
import os
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, average_precision_score, confusion_matrix

BASE_PATH = r"D:\Masters\Thesis\Block 3\Thesis work"
AUG_PATH = os.path.join(BASE_PATH, "Data", "augmented")
VAL_PATH = os.path.join(BASE_PATH, "Data", "processed", "dataset1", "D1_Val.csv")
REPORTS_PATH = os.path.join(BASE_PATH, "Reports")

os.makedirs(REPORTS_PATH, exist_ok=True)

candidate_files = [
    "D1_Train_UnderSampled.csv", "D1_Train_SMOTETomek.csv",
    "D1_Train_CTGAN_300.csv", "D1_Train_CTGAN_500.csv", "D1_Train_CTGAN_700.csv",
    "D1_Train_TVAEGAN300.csv", "D1_Train_TVAEGAN500.csv", "D1_Train_TVAEGAN700.csv"
]

def validate_basemodel(target_col):
    df_val = pd.read_csv(VAL_PATH)
    X_val = df_val.drop(columns=[target_col]).values
    y_val = df_val[target_col].values

    all_results = []

    for file_name in candidate_files:
        df_train = pd.read_csv(os.path.join(AUG_PATH, file_name))
        X_train = df_train.drop(columns=[target_col]).values
        y_train = df_train[target_col].values
        
        svm = SVC(kernel='rbf', probability=True, random_state=42)
        svm.fit(X_train, y_train)
        svm_probs = svm.predict_proba(X_val)[:, 1]
        svm_preds = svm.predict(X_val)

        xgb = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
        xgb.fit(X_train, y_train)
        xgb_probs = xgb.predict_proba(X_val)[:, 1]
        xgb_preds = xgb.predict(X_val)

        tn = TabNetClassifier(n_d=16, n_a=16, n_steps=5, mask_type='entmax', verbose=0)
        tn.fit(X_train=X_train, y_train=y_train, eval_set=[(X_val, y_val)], 
               max_epochs=100, patience=15, batch_size=1024, virtual_batch_size=128)
        tn_probs = tn.predict_proba(X_val)[:, 1]
        tn_preds = tn.predict(X_val)

        models = [("SVM", svm_probs, svm_preds), ("XGBoost", xgb_probs, xgb_preds), ("TabNet", tn_probs, tn_preds)]
        
        for name, probs, preds in models:
            all_results.append({
                "Augmentation": file_name.replace("D1_Train_", "").replace(".csv", ""),
                "Model": name,
                "Accuracy": round(accuracy_score(y_val, preds), 4),
                "F1-Score": round(f1_score(y_val, preds), 4),
                "PR-AUC": round(average_precision_score(y_val, probs), 4),
                "Precision":round(precision_score(y_val, preds), 4),
                "Recall":round(recall_score(y_val, preds), 4),
                "Confusion Matrix":str(confusion_matrix(y_val, preds).tolist())
            })

    results_df = pd.DataFrame(all_results)
    export_file = os.path.join(REPORTS_PATH, "D1_basemodel.csv")
    results_df.to_csv(export_file, index=False)
    print(results_df.sort_values(by="PR-AUC", ascending=False).head(10))

if __name__ == "__main__":
    validate_basemodel("status")
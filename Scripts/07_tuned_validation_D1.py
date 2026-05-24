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
    "D1_Train_TVAEGAN300.csv", "D1_Train_TVAEGAN500.csv", "D1_Train_TVAEGAN700.csv"]

svm_grid = []
for c in [0.05, 0.1, 0.5]:
    for g in [0.1, 1.0]:
        svm_grid.append({'id': f'SVM_C{c}_g{g}', 'C': c, 'gamma': g})
xgb_grid = []
for lr in [0.01, 0.1]:
    for sub in [0.8, 1.0]:
        xgb_grid.append({'id': f'XGB_L{lr}_S{sub}', 'lr': lr, 'subsample': sub})
tabnet_grid = []
for lamb in [1e-5, 5e-5, 1e-3]:
    for mom in [0.02, 0.05]:
        for m_type in ['entmax', 'sparsemax']:
            tabnet_grid.append({
                'id': f'TabNet_L{lamb}_M{mom}_{m_type}', 'lambda': lamb, 'mom': mom, 'mask': m_type})
def metrics(file_name, model_name, config_id, params, y_val, probs, preds):
    return{
            "Augmentation": file_name.replace("D1_Train_", "").replace(".csv", ""),
            "Model": model_name,
            "Config": config_id,
            "Params": params,
            "Accuracy": round(accuracy_score(y_val, preds), 4),
            "F1-Score": round(f1_score(y_val, preds), 4),
            "PR-AUC": round(average_precision_score(y_val, probs), 4),
            "Precision":round(precision_score(y_val, preds), 4),
            "Recall":round(recall_score(y_val, preds), 4),
            "Confusion Matrix":str(confusion_matrix(y_val, preds).tolist())}
def tuned_validation(target_col):
    df_val = pd.read_csv(VAL_PATH)
    X_val = df_val.drop(columns=[target_col]).values
    y_val = df_val[target_col].values
    all_results=[]

    for file_name in candidate_files:
        df_train = pd.read_csv(os.path.join(AUG_PATH, file_name))
        X_train = df_train.drop(columns=[target_col]).values
        y_train = df_train[target_col].values

        for p in svm_grid:
            model = SVC(kernel='rbf', C=p['C'], gamma=p['gamma'], probability=True, random_state=42)
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_val)[:, 1]
            preds = model.predict(X_val)
            all_results.append(metrics(file_name, "SVM", p['id'], f"C={p['C']}, g={p['gamma']}", y_val, probs, preds))

        for p in xgb_grid:
            model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=p['lr'], subsample=p['subsample'], random_state=42)
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_val)[:, 1]
            preds = model.predict(X_val)
            all_results.append(metrics(file_name, "XGBoost", p['id'], f"lr={p['lr']}, sub={p['subsample']}", y_val, probs, preds))

        for p in tabnet_grid:
            model = TabNetClassifier(n_d=16, n_a=16, n_steps=5, 
                                     mask_type=p['mask'], 
                                     lambda_sparse=p['lambda'], 
                                     momentum=p['mom'], 
                                     optimizer_params=dict(lr=0.02), verbose=0)
            model.fit(X_train=X_train, y_train=y_train, eval_set=[(X_val, y_val)], 
                      max_epochs=100, patience=15, batch_size=1024, virtual_batch_size=128)
            probs = model.predict_proba(X_val)[:, 1]
            preds = model.predict(X_val)
            p_str = f"L={p['lambda']}, M={p['mom']}, Mask={p['mask']}"
            all_results.append(metrics(file_name, "TabNet", p['id'], p_str, y_val, probs, preds))
    results_df = pd.DataFrame(all_results)
    export_file = os.path.join(REPORTS_PATH, "D1_tunedmodel.csv")
    results_df.to_csv(export_file, index=False)
    print(results_df.sort_values(by="PR-AUC", ascending=False).head(10))

if __name__ == "__main__":
    tuned_validation("status")
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier

BASE_PATH = r"D:\Masters\Thesis\Block 3\Thesis work"
D1_AUGMENTED_DIR = os.path.join(BASE_PATH, "Data", "augmented")
D1_PROCESSED_DIR = os.path.join(BASE_PATH, "Data", "processed", "dataset1")
REPORTS_DIR = os.path.join(BASE_PATH, "Reports")
os.makedirs(REPORTS_DIR, exist_ok=True)
TARGET = "status"
def xgboost_shap():
    df_train = pd.read_csv(os.path.join(D1_AUGMENTED_DIR, "D1_Train_CTGAN_300.csv"))
    df_test = pd.read_csv(os.path.join(D1_PROCESSED_DIR, "D1_Test.csv"))
    X_train = df_train.drop(columns=[TARGET])
    y_train = df_train[TARGET]
    X_test = df_test.drop(columns=[TARGET])
    feature_names = list(X_test.columns)
    dept_cols = [i for i, name in enumerate(feature_names) if "department" in name.lower()]
    other_cols = [i for i, name in enumerate(feature_names) if "department" not in name.lower()]
    model = XGBClassifier(
        n_estimators=100, 
        max_depth=4, 
        learning_rate=0.1, 
        subsample=1.0, 
        random_state=42)
    model.fit(X_train, y_train)
    explainer = shap.TreeExplainer(model)
    shap_values_raw = explainer(X_test)
    shap_raw_array = shap_values_raw.values
    if len(shap_raw_array.shape) == 3:
        shap_raw_array = shap_raw_array[:, :, 1]
    if dept_cols:
        dept_values = shap_raw_array[:, dept_cols].sum(axis=1, keepdims=True)
        other_values = shap_raw_array[:, other_cols]
        new_values = np.hstack([other_values, dept_values])
        dept_data = X_test.values[:, dept_cols].sum(axis=1, keepdims=True)
        other_data = X_test.values[:, other_cols]
        new_data = np.hstack([other_data, dept_data]) 
        new_feature_names = [feature_names[i] for i in other_cols] + ["Department"]
    else:
        new_values = shap_raw_array
        new_data = X_test.values
        new_feature_names = feature_names
    clean_shap_values = shap.Explanation(
        values=new_values,
        base_values=shap_values_raw.base_values,
        data=new_data,
        feature_names=new_feature_names)
    mean_abs_shap = np.abs(clean_shap_values.values).mean(axis=0).flatten()
    df_rank = pd.DataFrame({
        'Feature': new_feature_names,
        'Mean_Abs_SHAP': mean_abs_shap
    }).sort_values(by='Mean_Abs_SHAP', ascending=False)
    print(df_rank.to_string(index=False))
    plt.figure(figsize=(10, 8))
    shap.plots.bar(clean_shap_values, max_display=clean_shap_values.shape[1] + 1, show=False)
    ax = plt.gca()
    for text in list(ax.texts):
        text.remove()
    for patch in ax.patches:
        patch.set_facecolor("#008CA8")
        patch.set_edgecolor("#008CA8")   
    plt.tight_layout()
    output_path = os.path.join(REPORTS_DIR, "D1_SHAP_CTGAN_XGBoost.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
if __name__ == "__main__":
    xgboost_shap()
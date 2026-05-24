import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

CSV_SOURCE = r"D:\Masters\Thesis\Block 3\Thesis work\Reports\D1_no_hyperparameter.csv"
SAVE_DIR = r"D:\Masters\Thesis\Block 3\Thesis work\Graphs\dataset1\model analysis"
FILENAME = "01. CTGAN_bell_curve.png"

os.makedirs(SAVE_DIR, exist_ok=True)

if os.path.exists(CSV_SOURCE):
    df = pd.read_csv(CSV_SOURCE)
    ctgan_df = df[df['Augmentation'].str.contains('CTGAN', na=False)].copy()
    ctgan_df['Epochs'] = ctgan_df['Augmentation'].str.extract(r'(\d+)').astype(int)
    ctgan_df = ctgan_df.sort_values('Epochs')

    sns.set_theme(style="white")
    fig, axes = plt.subplots(3, 1, figsize=(8, 15), sharex=True)
    models = ['SVM', 'XGBoost', 'TabNet']
    metric = 'F1-Score'
    line_color = "#00000086" 

    for i, model in enumerate(models):
        model_data = ctgan_df[ctgan_df['Model'] == model]
        sns.lineplot(
            ax=axes[i],
            data=model_data,
            x='Epochs',
            y=metric,
            marker='o',
            linewidth=3,
            color=line_color
        )
        peak_row = model_data.loc[model_data[metric].idxmax()]
        axes[i].axvline(x=peak_row['Epochs'], color='grey', linestyle='--', alpha=0.5)
        axes[i].text(peak_row['Epochs']+10, 0.77, f"{peak_row['Epochs']} epochs", 
                     color='grey', fontsize=10)
        axes[i].grid(False)
        sns.despine(ax=axes[i], top=True, right=True)
        axes[i].set_title(f"{model}", fontsize=14)
        axes[i].set_ylabel("F1-Score", fontsize=11)
        axes[i].set_ylim(0.75, 1.0)
        axes[i].set_xticks([100, 200, 300, 500, 700])

    plt.suptitle("Training results on CTGAN", fontsize=16, y=1.01)
    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, FILENAME)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
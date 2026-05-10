# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, make_scorer, recall_score
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import LabelEncoder
import pandas as pd

# %%
def find_txt_files(training_root_path: Path) -> list[Path]:
    """Find all .txt files recursively under the training folder.
    Args:
            training_root_path: Root path for the training data directory.
    Returns:
            A sorted list of .txt file paths.
    """
    return sorted(training_root_path.rglob("*.txt"))


def parse_patient_txt_file(txt_file_path: Path) -> dict[str, str]:
    """Parse a patient text file of `Key: Value` rows.
    Args:
            txt_file_path: Path to one patient .txt file.
    Returns:
            A dictionary where keys are field names and values are raw string values.
    Raises:
            OSError: If the file cannot be read.
    """
    record: dict[str, str] = {}
    for raw_line in txt_file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", maxsplit=1)
        record[key.strip()] = value.strip()
    record["source_file"] = str(txt_file_path)
    return record


def normalize_string_missing_values(training_dataframe: pd.DataFrame) -> pd.DataFrame:
    """Normalize literal string missing markers to pandas missing values.
    Args:
            training_dataframe: Raw dataframe created from patient text files or parquet.
    Returns:
            A dataframe where common string missing markers are converted to ``pd.NA``.
    """
    normalized_dataframe = training_dataframe.copy()
    missing_markers = {
        "": pd.NA,
        "nan": pd.NA,
        "NaN": pd.NA,
        "NAN": pd.NA,
        "none": pd.NA,
        "None": pd.NA,
        "NONE": pd.NA,
        "null": pd.NA,
        "Null": pd.NA,
        "NULL": pd.NA,
    }
    return normalized_dataframe.replace(missing_markers)


def build_training_dataframe(training_root_path: Path) -> pd.DataFrame:
    """Load all patient text files into one pandas DataFrame.
    Args:
            training_root_path: Root path that contains patient folders and .txt files.
    Returns:
            A DataFrame with one row per file and one column per text field.
    """
    records: list[dict[str, str]] = []
    for txt_file_path in find_txt_files(training_root_path):
        records.append(parse_patient_txt_file(txt_file_path))
    training_dataframe = pd.DataFrame.from_records(records)
    training_dataframe = normalize_string_missing_values(training_dataframe)
    return convert_column_types(training_dataframe)


def convert_column_types(training_dataframe: pd.DataFrame) -> pd.DataFrame:
    """Convert known columns to analysis-friendly dtypes.
    Args:
            training_dataframe: DataFrame created from patient text files.
    Returns:
            A DataFrame with converted numeric and boolean columns.
    """
    converted_dataframe = normalize_string_missing_values(training_dataframe)
    for numeric_column in ["Age", "TTM", "CPC", "ROSC"]:
        if numeric_column in converted_dataframe.columns:
            converted_dataframe[numeric_column] = pd.to_numeric(
                converted_dataframe[numeric_column],
                errors="coerce",
            )
    for bool_column in ["OHCA", "Shockable Rhythm"]:
        if bool_column in converted_dataframe.columns:
            converted_dataframe[bool_column] = converted_dataframe[bool_column].map(
                {"True": True, "False": False}
            )
    return converted_dataframe


# %%
repo_root = Path.cwd()
if repo_root.name == "models":
    repo_root = repo_root.parent
training_root_path = repo_root / "icare_data" / "training"
training_dataframe = build_training_dataframe(training_root_path)

print(training_dataframe)

# %%
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score
import numpy as np

# ==========================================================
# 1. LOAD AND MERGE CNN OUT-OF-FOLD PREDICTIONS
# ==========================================================
# Adjust these paths to where you saved the new OOF CSVs
csv_b0_path = repo_root / "analysis" / "EfficientNet-B0" / "oof_predictions_cnn2_grid.csv"
csv_v2_path = repo_root / "analysis" / "EfficientNetV2-S_data" / "oof_predictions_cnn1_grid.csv"

cnn_df_b0 = pd.read_csv(csv_b0_path)
cnn_df_v2_s = pd.read_csv(csv_v2_path)

# Normalize CNN summaries and rename features by model
cnn_feature_cols = [
    'prob_poor_mean', 'prob_poor_std', 'prob_poor_min', 'prob_poor_max',
    'prob_poor_median', 'prob_poor_q25', 'prob_poor_q75',
    'prob_poor_count', 'prob_poor_frac_gt_05',
    'prob_poor_iqr', 'prob_poor_confidence',
]

for df, model_suffix in [(cnn_df_b0, 'b0'), (cnn_df_v2_s, 'v2_s')]:
    if 'prob_poor' in df.columns and 'prob_poor_mean' not in df.columns:
        df.rename(columns={'prob_poor': 'prob_poor_mean'}, inplace=True)

    available_cols = ['Patient'] + [c for c in cnn_feature_cols if c in df.columns]
    df = df[available_cols]
    df.rename(columns={c: f'{c}_{model_suffix}' for c in available_cols if c != 'Patient'}, inplace=True)
    if model_suffix == 'b0':
        cnn_df_b0 = df
    else:
        cnn_df_v2_s = df

# Merge CNNs together
cnn_patient_merged = pd.merge(cnn_df_b0, cnn_df_v2_s, on="Patient", how="inner")

# Merge with Clinical Metadata
training_dataframe["Patient"] = training_dataframe["source_file"].str.extract(r'(\d+)').astype(int)
merged_df = training_dataframe.merge(cnn_patient_merged, on="Patient", how="inner")

print("Merged Data Shape:", merged_df.shape)

# ==========================================================
# 2. FEATURE ENGINEERING & PREPROCESSING
# ==========================================================
model_df = merged_df.copy()

# Notice we removed prob_good and predicted_label (RF doesn't need redundant data)
cnn_base_features = [
    'prob_poor_mean', 'prob_poor_std', 'prob_poor_min', 'prob_poor_max',
    'prob_poor_median', 'prob_poor_q25', 'prob_poor_q75',
    'prob_poor_count', 'prob_poor_frac_gt_05',
    'prob_poor_iqr', 'prob_poor_confidence',
]
features = [
    "Age", "Sex", "ROSC", "OHCA", "Shockable Rhythm", "TTM",
] + [f"{col}_b0" for col in cnn_base_features] + [f"{col}_v2_s" for col in cnn_base_features]

# Drop missing targets
model_df = model_df.dropna(subset=["Outcome"])

# Encode target carefully: Ensure Good=0, Poor=1 (Matches CNN logic)
target_le = LabelEncoder()
y = target_le.fit_transform(model_df["Outcome"])
# Check mapping to be safe
print("Target Mapping:", dict(zip(target_le.classes_, target_le.transform(target_le.classes_))))

# Encode categorical features
categorical_cols = ["Sex", "OHCA", "Shockable Rhythm", "TTM"]
for col in categorical_cols:
    model_df[col] = model_df[col].astype(str)
    model_df[col] = LabelEncoder().fit_transform(model_df[col])

# Fill missing numericals with medians
model_df["Age"] = model_df["Age"].fillna(model_df["Age"].median())
model_df["ROSC"] = model_df["ROSC"].fillna(model_df["ROSC"].median())

X = model_df[features]

# ==========================================================
# 3. HYPERPARAMETER TUNING & NESTED CV
# ==========================================================
def sensitivity_at_95_specificity(y_true, y_probs):
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    idx = np.where(fpr <= 0.05)[0]
    if len(idx) == 0: return 0.0
    return tpr[idx[-1]]

tpr_at_fpr05_scorer = make_scorer(sensitivity_at_95_specificity, greater_is_better=True, response_method='predict_proba')

param_grid = {
    "n_estimators": [100],
    "max_depth": [None],
    "min_samples_split": [2],
    "min_samples_leaf": [2],
    "max_features": ["sqrt"],
}

# Use exactly the same CV strategy to prevent any alignment leaks
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
base_rf = RandomForestClassifier(random_state=1234, n_jobs=-1)

grid_search = GridSearchCV(
    estimator=base_rf,
    param_grid=param_grid,
    scoring=tpr_at_fpr05_scorer,
    cv=cv,
    n_jobs=-1
)

# ==========================================================
# 4. GENERATE CLEAN META-PREDICTIONS FOR ALL 607 PATIENTS
# ==========================================================
print("\nGenerating Stacked OOF Predictions for Random Forest...")
# cross_val_predict will run GridSearchCV internally for each fold, ensuring 0% data leakage!
rf_oof_probs = cross_val_predict(grid_search, X, y, cv=cv, method='predict_proba', n_jobs=-1)

# Probabilities for class '1' (Poor)
rf_prob_poor = rf_oof_probs[:, 1] 
rf_pred_class = (rf_prob_poor >= 0.5).astype(int)

# Attach results back to the dataframe
merged_df['rf_prob_poor'] = rf_prob_poor
merged_df['rf_predicted_label'] = target_le.inverse_transform(rf_pred_class)
merged_df['true_label_encoded'] = y

# Save Final Predictions
output_path = repo_root / "analysis" / "RF" / "final_ensemble_predictions_oof.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
merged_df.to_csv(output_path, index=False)
print(f"Saved unbiased Meta-Learner predictions to: {output_path}")

# ==========================================================
# 5. FINAL BOOTSTRAPPED PAPER METRICS (95% CI)
# ==========================================================
def calculate_metrics_with_ci(y_true, y_prob, n_bootstraps=1000, seed=1234):
    rng = np.random.RandomState(seed)
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)
    
    bootstrapped_aucs, bootstrapped_accs, bootstrapped_tprs = [], [], []
    
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_prob), len(y_prob))
        if len(np.unique(y_true[indices])) < 2: continue
            
        y_true_boot, y_prob_boot, y_pred_boot = y_true[indices], y_prob[indices], y_pred[indices]
        
        bootstrapped_aucs.append(roc_auc_score(y_true_boot, y_prob_boot))
        bootstrapped_accs.append(accuracy_score(y_true_boot, y_pred_boot))
        
        fpr, tpr, _ = roc_curve(y_true_boot, y_prob_boot)
        bootstrapped_tprs.append(np.interp(0.05, fpr, tpr))
        
    def get_ci(data): return np.mean(data), np.percentile(data, 2.5), np.percentile(data, 97.5)
        
    auc = get_ci(bootstrapped_aucs)
    acc = get_ci(bootstrapped_accs)
    tpr = get_ci(bootstrapped_tprs)
    return auc, acc, tpr

auc_ci, acc_ci, tpr_ci = calculate_metrics_with_ci(merged_df['true_label_encoded'], merged_df['rf_prob_poor'])

print("\n" + "="*50)
print("FINAL ENSEMBLE METRICS (N=607) WITH 95% CI")
print("="*50)
print(f"AUC:       {auc_ci[0]:.3f} (95% CI: {auc_ci[1]:.3f} - {auc_ci[2]:.3f})")
print(f"Accuracy:  {acc_ci[0]:.3f} (95% CI: {acc_ci[1]:.3f} - {acc_ci[2]:.3f})")
print(f"TPR@FPR05: {tpr_ci[0]:.3f} (95% CI: {tpr_ci[1]:.3f} - {tpr_ci[2]:.3f})")

# Feature Importances (Train on full dataset once just to see what was most important)
grid_search.fit(X, y)
importances = grid_search.best_estimator_.feature_importances_
print("\nFeature Importances:")
for name, imp in sorted(zip(features, importances), key=lambda x: x[1], reverse=True):
    print(f"  {name}: {imp:.4f}")

# %%
best_params = grid_search.best_params_
print(best_params)



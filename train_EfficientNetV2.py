from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split


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


def convert_column_types(training_dataframe: pd.DataFrame) -> pd.DataFrame:
    """Convert known columns to analysis-friendly dtypes.
    Args:
            training_dataframe: DataFrame created from patient text files.
    Returns:
            A DataFrame with converted numeric and boolean columns.
    """
    converted_dataframe = normalize_string_missing_values(training_dataframe)
    if "Patient" in converted_dataframe.columns:
        # Ensure patient ids align with folder naming like 0284.
        patient_series = converted_dataframe["Patient"].astype("string").str.strip()
        patient_series = patient_series.str.replace(r"\.0$", "", regex=True)
        converted_dataframe["Patient"] = patient_series.where(
            patient_series.isna(), patient_series.str.zfill(4)
        )
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


training_root_path = Path("icare_data") / "training"
training_dataframe = build_training_dataframe(training_root_path)

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def build_image_dataframe(
    patient_dataframe: pd.DataFrame, image_root_path: Path
) -> pd.DataFrame:
    """Expand patient-level labels into per-image samples.

    Args:
        patient_dataframe: Patient-level dataframe with `Patient` and `Outcome`.
        image_root_path: Root folder that contains one folder per patient.

    Returns:
        A dataframe with one row per image and columns for patient id, outcome,
        filepath, and numeric label.
    """
    label_map = {"Poor": 0, "Good": 1}
    image_records: list[dict[str, object]] = []

    for _, patient_row in patient_dataframe.iterrows():
        patient_id = str(patient_row["Patient"])
        outcome = patient_row["Outcome"]
        if pd.isna(outcome) or outcome not in label_map:
            continue

        patient_image_folder = image_root_path / patient_id
        if not patient_image_folder.is_dir():
            continue

        for image_path in sorted(patient_image_folder.glob("*.png")):
            image_records.append(
                {
                    "Patient": patient_id,
                    "Outcome": outcome,
                    "filepath": image_path,
                    "label": label_map[outcome],
                }
            )

    image_dataframe = pd.DataFrame.from_records(image_records)
    if not image_dataframe.empty:
        image_dataframe = image_dataframe.sort_values(
            ["Patient", "filepath"]
        ).reset_index(drop=True)
    return image_dataframe


# get all folder names in the training directory exports\mel_360x360\train
training_images_root = Path("exports") / "mel_360x360" / "train"
training_image_folders = sorted(training_images_root.iterdir())
print(f"Found {len(training_image_folders)} image folders in {training_images_root}")

patients_with_images = set(
    folder.name for folder in training_image_folders if folder.is_dir()
)
print(f"Patients with images: {len(patients_with_images)}")

# Patient-level split for model training and evaluation.
split_df = training_dataframe.dropna(subset=["Outcome"]).copy()

train_df, test_df = train_test_split(
    split_df,
    test_size=0.2,
    stratify=split_df["Outcome"],
    random_state=1234,
)

train_df["has_images"] = train_df["Patient"].isin(patients_with_images)
test_df["has_images"] = test_df["Patient"].isin(patients_with_images)

train_df_with_images = train_df[train_df["has_images"]].copy()
test_df_with_images = test_df[test_df["has_images"]].copy()

patient_level_train_df = train_df_with_images.drop_duplicates(subset=["Patient"]).copy()
patient_level_train_df = patient_level_train_df.sort_values(
    ["Outcome", "Patient"]
).reset_index(drop=True)

validation_patient_count = max(2, round(len(patient_level_train_df) * 0.15))
validation_patient_count = min(
    validation_patient_count, len(patient_level_train_df) - 1
)

if patient_level_train_df["Outcome"].value_counts().min() < 2:
    train_patient_df = patient_level_train_df.copy()
    val_patient_df = patient_level_train_df.iloc[0:0].copy()
else:
    train_patient_df, val_patient_df = train_test_split(
        patient_level_train_df,
        test_size=validation_patient_count,
        stratify=patient_level_train_df["Outcome"],
        random_state=415,
    )

train_image_dataframe = build_image_dataframe(train_patient_df, training_images_root)
val_image_dataframe = build_image_dataframe(val_patient_df, training_images_root)
test_image_dataframe = build_image_dataframe(
    test_df_with_images.drop_duplicates(subset=["Patient"]), training_images_root
)

print("Train patients:", train_patient_df.shape[0])
print("Validation patients:", val_patient_df.shape[0])
print(
    "Test patients:", test_df_with_images.drop_duplicates(subset=["Patient"]).shape[0]
)
print("Train images:", len(train_image_dataframe))
print("Validation images:", len(val_image_dataframe))
print("Test images:", len(test_image_dataframe))

print("Train image label distribution:")
print(
    f"{train_image_dataframe['Outcome'].value_counts()} ({ train_image_dataframe['Outcome'].value_counts()/len(train_image_dataframe) })"
)
print("Validation image label distribution:")
print(
    f"{val_image_dataframe['Outcome'].value_counts()} ({ val_image_dataframe['Outcome'].value_counts()/len(val_image_dataframe)  })"
)
print("Test image label distribution:")
print(
    f"{test_image_dataframe['Outcome'].value_counts()} ({ test_image_dataframe['Outcome'].value_counts()/len(test_image_dataframe)  })"
)


def to_relative_paths(
    image_dataframe: pd.DataFrame, image_root_path: Path
) -> pd.DataFrame:
    """Convert absolute file paths to relative paths for cross-platform portability.
    Args:
        image_dataframe: Dataframe containing a `filepath` column.
        image_root_path: Root folder used to create relative image paths.
    Returns:
        A copy of the dataframe with portable relative paths.
    """
    portable_dataframe = image_dataframe.copy()
    portable_dataframe["relative_path"] = portable_dataframe["filepath"].map(
        lambda path_value: str(Path(path_value).relative_to(image_root_path))
    )
    return portable_dataframe


artifact_root = Path("artifacts") / "splits"
artifact_root.mkdir(parents=True, exist_ok=True)

train_artifact_df = to_relative_paths(train_image_dataframe, training_images_root)
val_artifact_df = to_relative_paths(val_image_dataframe, training_images_root)
test_artifact_df = to_relative_paths(test_image_dataframe, training_images_root)

artifact_columns = ["Patient", "Outcome", "label", "relative_path"]
train_artifact_df[artifact_columns].to_parquet(
    artifact_root / "train_split.parquet", index=False
)
val_artifact_df[artifact_columns].to_parquet(
    artifact_root / "val_split.parquet", index=False
)
test_artifact_df[artifact_columns].to_parquet(
    artifact_root / "test_split.parquet", index=False
)

print(f"Saved split artifacts to {artifact_root.resolve()}")

# Local prep notebook: training has been moved to a dedicated Colab notebook.
# This cell keeps a quick integrity check before upload to Google Drive.
artifact_root = Path("artifacts") / "splits"

required_files = [
    artifact_root / "train_split.parquet",
    artifact_root / "val_split.parquet",
    artifact_root / "test_split.parquet",
]

for required_file in required_files:
    print(required_file, "exists=", required_file.exists())

train_patients = set(
    pd.read_parquet(artifact_root / "train_split.parquet")["Patient"]
    .astype(str)
    .unique()
)
val_patients = set(
    pd.read_parquet(artifact_root / "val_split.parquet")["Patient"].astype(str).unique()
)
test_patients = set(
    pd.read_parquet(artifact_root / "test_split.parquet")["Patient"]
    .astype(str)
    .unique()
)

print("train/val overlap:", len(train_patients & val_patients))
print("train/test overlap:", len(train_patients & test_patients))
print("val/test overlap:", len(val_patients & test_patients))

# Intentionally left as local-prep completion marker.
# Use train_EfficientNetV2_colab.ipynb for training, evaluation, and model saving.
print(
    "Local preparation complete. Upload artifacts/splits/*.parquet and images to Google Drive for Colab training."
)

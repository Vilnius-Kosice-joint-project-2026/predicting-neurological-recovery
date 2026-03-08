from pathlib import Path

import pandas as pd


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
    return convert_column_types(training_dataframe)


def convert_column_types(training_dataframe: pd.DataFrame) -> pd.DataFrame:
    """Convert known columns to analysis-friendly dtypes.

    Args:
            training_dataframe: DataFrame created from patient text files.

    Returns:
            A DataFrame with converted numeric and boolean columns.
    """
    converted_dataframe = training_dataframe.copy()

    for numeric_column in ["Age", "TTM", "CPC"]:
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

    if "ROSC" in converted_dataframe.columns:
        converted_dataframe["ROSC"] = pd.to_numeric(
            converted_dataframe["ROSC"],
            errors="coerce",
        )

    return converted_dataframe


def main() -> None:
    """Build and print training DataFrame details."""
    training_root_path = Path("icare_data") / "training"
    training_dataframe = build_training_dataframe(training_root_path)

    print("Loaded rows:", len(training_dataframe))
    print("Columns:", list(training_dataframe.columns))
    print(training_dataframe.head())


if __name__ == "__main__":
    main()

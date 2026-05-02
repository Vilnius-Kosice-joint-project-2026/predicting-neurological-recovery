import pandas as pd
import shutil
import os


def write_patient_file(output_dir, patient_id, outcome, probability, cpc=3.0):
    patient_folder = os.path.join(output_dir, patient_id)
    os.makedirs(patient_folder, exist_ok=True)
    output_path = os.path.join(patient_folder, f"{patient_id}.txt")

    with open(output_path, "w") as f:
        f.write(f"Patient: {patient_id}\n")
        f.write(f"Outcome: {outcome}\n")
        f.write(f"Outcome Probability: {probability:.3f}\n")
        f.write(f"CPC: {cpc:.3f}\n")

    return output_path


if __name__ == "__main__":

    output_dir = os.path.join(
        "analysis", "official_scoring_metric", "demo_data", "outputs_generated"
    )

    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(
        "analysis\\RF\\random_forest_predictions.csv",
        dtype={"Patient": str},
    )
    df['Patient'] = df['Patient'].str.zfill(4)

    # read all folder names in icare_data\training
    training_patients_df = pd.read_csv("artifacts//combined_patient_data.csv", dtype={"Patient": str})
    training_patients_df['Patient'] = training_patients_df['Patient'].str.zfill(4)
    training_patients = set(training_patients_df["Patient"].tolist())

    print(training_patients_df.head())

    # create a set of patients that are in df['Patient'].unique() and in training_patients
    patients_in_both = set(df['Patient'].unique()) & training_patients

    print(f"Patients in both: {patients_in_both}")
    print(f"Number of patients in both: {len(patients_in_both)}")

    # filter out from df any patients that are not in training_patients
    df = df[df['Patient'].isin(training_patients)]

    patient_preds = (
        df
        .groupby("Patient", as_index=False)[["rf_proba_0", "rf_proba_1"]]
        .mean()
    )
    patient_preds["final_prediction"] = (patient_preds["rf_proba_1"] > 0.5).astype(int)
    patient_preds["Outcome"] = patient_preds["final_prediction"].map({0: "Good", 1: "Poor"})

    for _, row in patient_preds.sort_values("Patient").iterrows():
        outcome = row["Outcome"]
        probability = row["rf_proba_0"] if outcome == "Good" else row["rf_proba_1"]
        cpc = 1.0 if outcome == "Good" else 5.0

        write_patient_file(output_dir, row["Patient"], outcome, probability, cpc)

    print(f"Wrote {len(patient_preds)} patient files to {output_dir}")

    # take all unique patients fromm df and copy their original data file from icare_data\training\{patient_id}\{patient_id}.txt to analysis\official_scoring_metric\demo_data\labels_generated
    labels_output_dir = os.path.join(
        "analysis", "official_scoring_metric", "demo_data", "labels_generated"
    )

    shutil.rmtree(labels_output_dir, ignore_errors=True)
    os.makedirs(labels_output_dir, exist_ok=True)

    for patient_id in df['Patient'].unique():
        src_path = os.path.join("icare_data", "training", patient_id, f"{patient_id}.txt")
        patient_label_dir = os.path.join(labels_output_dir, patient_id)
        os.makedirs(patient_label_dir, exist_ok=True)
        dst_path = os.path.join(patient_label_dir, f"{patient_id}.txt")
        shutil.copy(src_path, dst_path)

    print(f"Copied {len(df['Patient'].unique())} patient label files to {labels_output_dir}")

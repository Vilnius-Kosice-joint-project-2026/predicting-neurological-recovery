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

    # 1. Load the new OOF predictions CSV
    df = pd.read_csv(
        os.path.join("analysis", "RF", "final_ensemble_predictions_oof.csv"),
        dtype={"Patient": str},
    )
    df['Patient'] = df['Patient'].str.zfill(4)

    # 2. Read all folder names in icare_data/training
    training_patients_df = pd.read_csv(
        os.path.join("artifacts", "combined_patient_data.csv"), 
        dtype={"Patient": str}
    )
    training_patients_df['Patient'] = training_patients_df['Patient'].str.zfill(4)
    training_patients = set(training_patients_df["Patient"].tolist())

    print(f"Total patients in combined_patient_data.csv: {len(training_patients)}")

    # 3. Create a set of patients that are in df['Patient'].unique() and in training_patients
    patients_in_both = set(df['Patient'].unique()) & training_patients

    print(f"Number of patients in both: {len(patients_in_both)}")

    # 4. Filter out from df any patients that are not in training_patients
    df = df[df['Patient'].isin(training_patients)]

    # Because our new OOF dataframe already has exactly 1 row per patient, 
    # we don't need to groupby().mean() anymore. We can just iterate directly.
    
    patient_preds = df[['Patient', 'rf_prob_poor']].copy()
    
    # Generate the thresholded final prediction (>= 0.5 means "Poor" because Poor=1)
    patient_preds["final_prediction"] = (patient_preds["rf_prob_poor"] >= 0.5).astype(int)
    patient_preds["Outcome"] = patient_preds["final_prediction"].map({0: "Good", 1: "Poor"})

    # 5. Write the Challenge .txt files
    for _, row in patient_preds.sort_values("Patient").iterrows():
        outcome = row["Outcome"]
        
        # If predicted Poor, probability is rf_prob_poor. If Good, it's (1 - rf_prob_poor)
        if outcome == "Poor":
            probability = row["rf_prob_poor"]
            cpc = 5.0
        else:
            probability = 1.0 - row["rf_prob_poor"]
            cpc = 1.0

        write_patient_file(output_dir, row["Patient"], outcome, probability, cpc)

    print(f"Wrote {len(patient_preds)} patient files to {output_dir}")

    # 6. Copy original label data
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
        
        # Only copy if the source file actually exists to prevent crashes
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)

    print(f"Copied {len(df['Patient'].unique())} patient label files to {labels_output_dir}")
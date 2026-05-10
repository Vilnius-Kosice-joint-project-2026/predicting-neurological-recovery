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
    # Define directories
    output_dir = os.path.join(
        "analysis", "official_scoring_metric", "demo_data", "outputs_generated"
    )
    labels_output_dir = os.path.join(
        "analysis", "official_scoring_metric", "demo_data", "labels_generated"
    )

    # Clean up old directories
    for d in [output_dir, labels_output_dir]:
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)

    # 1. Load the new OOF CNN predictions
    # cnn_csv_path = os.path.join("analysis", "EfficientNetV2-S_data", "oof_predictions_cnn1_grid.csv")
    cnn_csv_path = os.path.join("analysis", "EfficientNet-B0", "oof_predictions_cnn2_grid.csv")
    
    df = pd.read_csv(cnn_csv_path, dtype={"Patient": str})
    df['Patient'] = df['Patient'].str.zfill(4)

    # 2. Read metadata to ensure patient alignment
    training_patients_df = pd.read_csv(
        os.path.join("artifacts", "combined_patient_data.csv"), 
        dtype={"Patient": str}
    )
    training_patients_df['Patient'] = training_patients_df['Patient'].str.zfill(4)
    training_patients = set(training_patients_df["Patient"].tolist())

    # 3. Filter to ensure only patients in the metadata are processed
    df = df[df['Patient'].isin(training_patients)]
    
    # 4. Prepare predictions
    # If your OOF file has multiple images per patient, mean() is required. 
    # If it is already 1 row per patient, this won't change anything.
    patient_preds = (
        df.groupby("Patient", as_index=False)[["prob_poor"]]
        .mean()
    )

    # 5. Generate Challenge files
    for _, row in patient_preds.sort_values("Patient").iterrows():
        p_poor = row["prob_poor"]
        
        # The official scorer ALWAYS expects the probability of a Poor outcome
        probability = p_poor
        
        # Decision logic: Outcome is Poor if prob_poor >= 0.5
        if p_poor >= 0.5:
            outcome = "Poor"
            cpc = 5.0
        else:
            outcome = "Good"
            cpc = 1.0

        write_patient_file(output_dir, row["Patient"], outcome, probability, cpc)

    print(f"Wrote {len(patient_preds)} CNN prediction files to {output_dir}")

    # 6. Copy original labels for the metric calculator
    for patient_id in patient_preds['Patient'].unique():
        src_path = os.path.join("icare_data", "training", patient_id, f"{patient_id}.txt")
        
        if os.path.exists(src_path):
            patient_label_dir = os.path.join(labels_output_dir, patient_id)
            os.makedirs(patient_label_dir, exist_ok=True)
            dst_path = os.path.join(patient_label_dir, f"{patient_id}.txt")
            shutil.copy(src_path, dst_path)

    print(f"Copied {len(patient_preds)} label files to {labels_output_dir}")
import os
import random
import re
import pickle
import lmdb
import mne
import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Optional, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- Configuration ---
DATA_ROOT = Path("icare_data/training")
OUTPUT_DB_DIR = Path("icare_data/processed_mamba")
SAMPLING_RATE = 200
WINDOW_SECONDS = 10
SEED = 42
NUM_CORES = 6
CHANNELS = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz"
]

# Filtering parameters
LOW_CUTOFF = 0.3
HIGH_CUTOFF = 75.0
NOTCH_FREQ = 60.0

def normalize_channel_label(channel_name: str) -> str:
    """Normalize EEG channel names into canonical 10-20 labels."""
    compact_name = channel_name.strip().replace(" ", "")
    normalized = compact_name.upper()
    alias_candidates = {
        "FP1": "Fp1", "FP2": "Fp2", "F3": "F3", "F4": "F4",
        "C3": "C3", "C4": "C4", "P3": "P3", "P4": "P4",
        "O1": "O1", "O2": "O2", "F7": "F7", "F8": "F8",
        "T3": "T3", "T4": "T4", "T5": "T5", "T6": "T6",
        "T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6",
        "FZ": "Fz", "CZ": "Cz", "PZ": "Pz",
    }
    return alias_candidates.get(normalized, compact_name)

def get_outcome(patient_dir: Path) -> Optional[int]:
    """Read patient metadata and return binary outcome."""
    patient_id = patient_dir.name
    txt_path = patient_dir / f"{patient_id}.txt"
    if not txt_path.exists():
        return None
    try:
        content = txt_path.read_text()
        match = re.search(r"Outcome: (Good|Poor)", content)
        if match:
            return 0 if match.group(1) == "Good" else 1
    except Exception:
        return None
    return None

def load_icare_segment(hea_path: Path, mat_path: Path, duration_seconds: Optional[float] = None) -> Optional[np.ndarray]:
    """Load and calibrate I-CARE EEG segment, optionally cropping to duration."""
    try:
        lines = hea_path.read_text().splitlines()
        first_tokens = lines[0].split()
        fs = float(first_tokens[2])
        
        channel_names = []
        gains = []
        baselines = []
        
        for line in lines[1:]:
            if not line.strip() or line.startswith("#"): break
            parts = line.split()
            if len(parts) < 3: continue
            
            gain_baseline_match = re.match(r"([\d\.eE+\-]+)\(([+\-]?\d+)\)", parts[2])
            if gain_baseline_match:
                gains.append(float(gain_baseline_match.group(1)))
                baselines.append(int(gain_baseline_match.group(2)))
            else:
                gains.append(1.0)
                baselines.append(0)
            channel_names.append(normalize_channel_label(parts[-1]))

        mat_data = scipy.io.loadmat(str(mat_path), squeeze_me=True)
        raw_signal = None
        for key in mat_data:
            if isinstance(mat_data[key], np.ndarray) and mat_data[key].ndim == 2:
                raw_signal = mat_data[key]
                break
        
        if raw_signal is None: return None
        if raw_signal.shape[0] != len(channel_names): raw_signal = raw_signal.T
        
        calibrated = (raw_signal - np.array(baselines)[:, None]) / np.array(gains)[:, None]
        
        # Standardize to 19 channels
        standardized = np.full((len(CHANNELS), calibrated.shape[1]), np.nan)
        for i, name in enumerate(CHANNELS):
            if name in channel_names:
                standardized[i, :] = calibrated[channel_names.index(name), :]
        
        # Fill NaNs with 0 (simple baseline)
        standardized = np.nan_to_num(standardized)
        
        # Create MNE Raw object for filtering
        info = mne.create_info(ch_names=CHANNELS, sfreq=fs, ch_types='eeg')
        raw = mne.io.RawArray(standardized, info, verbose=False)
        
        # Filtering (Apply to full signal first to avoid length warnings and artifacts)
        raw.filter(LOW_CUTOFF, HIGH_CUTOFF, fir_design='firwin', verbose=False)
        raw.notch_filter(NOTCH_FREQ, fir_design='firwin', verbose=False)
        
        # Crop to duration if specified
        if duration_seconds is not None:
            if raw.times[-1] < duration_seconds:
                return None
            raw.crop(tmin=0, tmax=duration_seconds, include_tmax=False)
        
        # Resampling
        raw.resample(SAMPLING_RATE, verbose=False)
        
        return raw.get_data()
    except Exception as e:
        print(f"Error loading {hea_path.name}: {e}")
        return None

def process_patient_worker(patient_dir: Path, label: int) -> List[Tuple[str, Dict]]:
    """Worker function to process all hours for a single patient."""
    samples = []
    # Find EEG segments
    hea_files = sorted(list(patient_dir.glob("*_EEG.hea")))
    for hea_path in hea_files:
        mat_path = hea_path.with_suffix(".mat")
        if not mat_path.exists(): continue
        
        data = load_icare_segment(hea_path, mat_path, duration_seconds=WINDOW_SECONDS)
        if data is None: continue
        
        # Reshape to (channels, WINDOW_SECONDS, SAMPLING_RATE)
        try:
            window = data.reshape(len(CHANNELS), WINDOW_SECONDS, SAMPLING_RATE)
        except ValueError:
            continue
            
        # Normalize by 100 (standard for this pipeline)
        window = window / 100.0
        
        sample_key = f"{hea_path.stem}_0" # Only one window per hour
        data_dict = {'sample': window.astype(np.float32), 'label': label}
        samples.append((sample_key, data_dict))
    return samples

def main():
    if not OUTPUT_DB_DIR.exists():
        OUTPUT_DB_DIR.mkdir(parents=True)
    
    db = lmdb.open(str(OUTPUT_DB_DIR), map_size=int(4e9)) # 4GB map size
    dataset_keys = {'train': [], 'val': [], 'test': []}
    
    # Only include directories that actually contain EEG data and have a valid outcome
    all_dirs = sorted([p for p in DATA_ROOT.iterdir() if p.is_dir()])
    
    valid_patients: List[Dict] = []
    for p in all_dirs:
        if any(p.glob("*_EEG.hea")):
            outcome = get_outcome(p)
            if outcome is not None:
                valid_patients.append({'path': p, 'outcome': outcome})
    
    n = len(valid_patients)
    print(f"Total directories found: {len(all_dirs)}")
    print(f"Valid patients with EEG and labels: {n}")
    
    if n == 0:
        print("No valid patient data found. Check DATA_ROOT path.")
        return

    # Extract paths and outcomes for stratification
    patient_paths = [p['path'] for p in valid_patients]
    outcomes = [p['outcome'] for p in valid_patients]

    # Stratified split: 70% train, 15% val, 15% test
    # First split: 70% train, 30% temp (val + test)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        patient_paths, outcomes, test_size=0.3, stratify=outcomes, random_state=SEED
    )
    
    # Second split: split temp into 50% val, 50% test (15% each of total)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.5, stratify=temp_labels, random_state=SEED
    )

    splits = {
        'train': train_paths,
        'val': val_paths,
        'test': test_paths
    }
    
    split_labels = {
        'train': list(train_labels),
        'val': list(val_labels),
        'test': list(test_labels)
    }

    # Print distribution summary
    for name in ['train', 'val', 'test']:
        labels = split_labels[name]
        n_good = labels.count(0)
        n_poor = labels.count(1)
        print(f"{name.capitalize()} split: {len(labels)} patients (Good: {n_good}, Poor: {n_poor})")

    for split_name, dirs in splits.items():
        print(f"Processing {split_name} split...")
        labels_list = split_labels[split_name]
        
        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
            futures = [
                executor.submit(process_patient_worker, d, l) 
                for d, l in zip(dirs, labels_list)
            ]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Split: {split_name}"):
                patient_samples = future.result()
                for sample_key, data_dict in patient_samples:
                    with db.begin(write=True) as txn:
                        txn.put(sample_key.encode(), pickle.dumps(data_dict))
                    dataset_keys[split_name].append(sample_key)

    with db.begin(write=True) as txn:
        txn.put('__keys__'.encode(), pickle.dumps(dataset_keys))
    
    db.close()
    print("Preprocessing complete.")

if __name__ == "__main__":
    main()

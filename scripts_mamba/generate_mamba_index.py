import lmdb
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

# --- Configuration ---
DB_DIR = Path("icare_data/processed_mamba")
SEED = 42
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15

def generate_index():
    if not DB_DIR.exists():
        print(f"Error: Database directory {DB_DIR} not found.")
        return

    print(f"Opening LMDB at {DB_DIR}...")
    db = lmdb.open(str(DB_DIR), readonly=False, lock=True)

    all_keys = []
    labels = []

    print("Scanning database for keys and labels...")
    with db.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            key_str = key.decode()
            if key_str == "__keys__":
                continue
            
            # Load the data to get the label
            try:
                data = pickle.loads(value)
                all_keys.append(key_str)
                labels.append(data['label'])
            except Exception as e:
                print(f"Skipping key {key_str} due to error: {e}")

    n_samples = len(all_keys)
    if n_samples == 0:
        print("No samples found in the database.")
        db.close()
        return

    print(f"Found {n_samples} samples.")
    n_good = labels.count(0)
    n_poor = labels.count(1)
    print(f"Distribution: Good={n_good}, Poor={n_poor}")

    # Perform Stratified Split
    # 1. Split Train and Temp (Val + Test)
    train_keys, temp_keys, train_labels, temp_labels = train_test_split(
        all_keys, labels, 
        test_size=(VAL_SIZE + TEST_SIZE), 
        stratify=labels, 
        random_state=SEED
    )

    # 2. Split Temp into Val and Test
    val_keys, test_keys, val_labels, test_labels = train_test_split(
        temp_keys, temp_labels, 
        test_size=0.5, # Assuming 15%/15% split
        stratify=temp_labels, 
        random_state=SEED
    )

    dataset_keys = {
        'train': train_keys,
        'val': val_keys,
        'test': test_keys
    }

    print("\nSplit Summary:")
    for split, keys in dataset_keys.items():
        print(f"  {split.capitalize()}: {len(keys)} samples")

    # Save to LMDB
    print("\nSaving __keys__ index to LMDB...")
    with db.begin(write=True) as txn:
        txn.put('__keys__'.encode(), pickle.dumps(dataset_keys))

    db.close()
    print("Done! You can now proceed with training.")

if __name__ == "__main__":
    generate_index()

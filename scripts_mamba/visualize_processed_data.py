import lmdb
import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
from pathlib import Path

# --- Configuration ---
PROCESSED_DB_DIR = Path("icare_data/processed_mamba")
CHANNELS = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz"
]

def visualize_random_sample():
    if not PROCESSED_DB_DIR.exists():
        print(f"Error: Database directory {PROCESSED_DB_DIR} not found.")
        return

    db = lmdb.open(str(PROCESSED_DB_DIR), readonly=True, lock=False)
    
    with db.begin() as txn:
        keys_data = txn.get('__keys__'.encode())
        if not keys_data:
            print("Error: Could not find keys in database.")
            return
        
        dataset_keys = pickle.loads(keys_data)
        all_keys = dataset_keys['train'] + dataset_keys['val'] + dataset_keys['test']
        
        if not all_keys:
            print("Error: No samples found in database.")
            return
        
        # Pick a random sample
        random_key = random.choice(all_keys)
        sample_data = txn.get(random_key.encode())
        sample_dict = pickle.loads(sample_data)
        
        data = sample_dict['sample'] # Shape: (19, 30, 200)
        label = sample_dict['label']
        
        print(f"Visualizing Sample: {random_key}")
        print(f"Label: {'Poor Outcome' if label == 1 else 'Good Outcome'}")
        print(f"Data Shape: {data.shape}")
        
        # Flatten for plotting: (19, 30*200) -> (19, 6000)
        flattened_data = data.reshape(19, -1)
        time_axis = np.linspace(0, 30, flattened_data.shape[1])
        
        plt.figure(figsize=(15, 12))
        spacing = 1.0 # Since data is normalized to ~[-1, 1], 1.0 spacing is visible
        
        for i in range(19):
            plt.plot(time_axis, flattened_data[i] - i * spacing, label=CHANNELS[i], linewidth=0.7)
            
        plt.yticks([-i * spacing for i in range(19)], CHANNELS)
        plt.xlabel("Time (seconds)")
        plt.title(f"Processed EEG Sample: {random_key} ({'Poor' if label == 1 else 'Good'} Outcome)")
        plt.grid(True, alpha=0.3)
        
        # Save visualization to artifacts or show
        output_path = Path("icare_data/sample_visualization.png")
        plt.savefig(output_path)
        print(f"Visualization saved to {output_path}")
        plt.close()

    db.close()

if __name__ == "__main__":
    visualize_random_sample()

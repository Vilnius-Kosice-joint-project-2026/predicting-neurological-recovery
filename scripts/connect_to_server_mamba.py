import posixpath
import paramiko
import stat
import os
import re
import lmdb
import pickle
from dotenv import load_dotenv
import time
from workflow_mne_mamba import do_everything

load_dotenv()
hostname = os.getenv("HOSTNAME")
port = os.getenv("PORT")
username = os.getenv("HPC_USERNAME")
private_key_path = os.getenv("PRIVATE_KEY_PATH")
remote_path = os.getenv("REMOTE_PATH")
OUTPUT_DB_DIR = "icare_data/processed_mamba"

VALID_EEG_RECORD_PATTERN = re.compile(r"^\d{4}_\d{3}_\d{3}_EEG\.(?:hea|mat)$")


def is_valid_eeg_record_filename(filename: str) -> bool:
    """Return True only for canonical WFDB EEG record files."""
    return bool(VALID_EEG_RECORD_PATTERN.match(filename))

def main():
    try:
        # Initialize SSH Client
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # Load the private key
        # Use RSAKey, Ed25519Key, or the generic from_private_key_file
        key = paramiko.RSAKey.from_private_key_file(private_key_path)

        print(f"Connecting to {hostname}...")
        client.connect(hostname, port, username, pkey=key)
        client.get_transport().set_keepalive(60)

        # Open SFTP session to list folders
        sftp = client.open_sftp()
        
        # Initialize LMDB environment (shared across the process)
        if not os.path.exists(OUTPUT_DB_DIR):
            os.makedirs(OUTPUT_DB_DIR, exist_ok=True)
        # Open in write mode so we can both check existence and write later
        db = lmdb.open(OUTPUT_DB_DIR, map_size=int(10e9)) 


        print(f"Retrieving folders in: {remote_path}\n")

        # List directory and filter for directories only
        contents = sftp.listdir_attr(remote_path)
        folders = [item.filename for item in contents if stat.S_ISDIR(item.st_mode)]

        if folders:
            print(f"Found {len(folders)} folders:")
            print(folders)

            

            for folder in sorted(folders):
                try:
                    folder_path = posixpath.join(remote_path, folder)
                    print(f"\nListing files in {folder_path} containing 'EEG':")

                    try:
                        folder_contents = sftp.listdir_attr(folder_path)
                        eeg_files = sorted(
                            item.filename
                            for item in folder_contents
                            if not stat.S_ISDIR(item.st_mode) 
                            and is_valid_eeg_record_filename(item.filename)
                            and int(item.filename.split('_')[1]) <= 2
                        )

                        if eeg_files:
                            print(eeg_files)

                            import glob
                            for eeg_file in eeg_files:
                                # eeg_file example: 0299_001_021_EEG.mat or .hea
                                parts = eeg_file.replace(".mat", "").replace(".hea", "").split('_')
                                if len(parts) >= 4:
                                    segment_token = f"{parts[0]}_{parts[1]}_{parts[2]}_{parts[3]}"
                                else:
                                    segment_token = eeg_file.split(".")[0]
                                
                                # Key example: 0299_001_021_EEG_0
                                segment_token_mamba = f"{segment_token}_0"
                                
                                is_processed = False
                                with db.begin() as txn:
                                    if txn.get(segment_token_mamba.encode()):
                                        is_processed = True
                                
                                if is_processed:
                                    print(f"Data already exists in LMDB for {eeg_file} (Key: {segment_token_mamba})")
                                else:
                                    print(f"No entry found in LMDB for {eeg_file}. Downloading...")
                                    
                                    # Also ensure the .txt metadata file is downloaded for the patient
                                    txt_file = f"{folder}.txt"
                                    local_txt_path = f"icare_data/training/{folder}/{txt_file}"
                                    if not os.path.exists(local_txt_path):
                                        remote_txt_path = posixpath.join(folder_path, txt_file)
                                        try:
                                            print(f"Downloading metadata {remote_txt_path}...")
                                            sftp.get(remote_txt_path, local_txt_path)
                                        except Exception as txt_e:
                                            print(f"Warning: Could not download metadata {txt_file}: {txt_e}")

                                    # Download the EEG file
                                    local_file_path = f"icare_data/training/{folder}/{eeg_file}"
                                    if os.path.exists(local_file_path):
                                        print(f"EEG file {eeg_file} already exists locally.")
                                    else:
                                        remote_file_path = posixpath.join(folder_path, eeg_file)
                                        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                                        print(f"Downloading {remote_file_path} to {local_file_path}...")
                                        tmp_file_path = local_file_path + ".tmp"
                                        sftp.get(remote_file_path, tmp_file_path)
                                        os.rename(tmp_file_path, local_file_path)
                                        time.sleep(0.5)

                        else:
                            print("  No EEG files found in this folder.")
                    except IOError as io_err:
                        print(f"  Could not list {folder_path}: {io_err}")

                    print("doing everything for folder:", folder)
                    print(folder)
                    do_everything(folder, db=db)


                    # Delete the downloaded raw files to save disk space
                    local_folder_path = os.path.join("icare_data", "training", folder)
                    if os.path.exists(local_folder_path):
                        import shutil
                        print(f"Cleaning up raw data in {local_folder_path} to conserve disk space...")
                        shutil.rmtree(local_folder_path)
                
                except Exception as loop_e:
                    print(f"Error processing folder {folder}: {loop_e}")
                    continue

        else:
            print("No folders found or path is empty.")

        db.close()
        sftp.close()
        client.close()

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

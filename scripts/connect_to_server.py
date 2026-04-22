import posixpath
import paramiko
import stat
import os
from dotenv import load_dotenv
import time
from workflow_mne import do_everything

load_dotenv()
hostname = os.getenv("HOSTNAME")
port = os.getenv("PORT")
username = os.getenv("HPC_USERNAME")
private_key_path = os.getenv("PRIVATE_KEY_PATH")
remote_path = os.getenv("REMOTE_PATH")

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
                            and "EEG" in item.filename
                            and len(item.filename.split('_')) > 2
                            and int(item.filename.split('_')[1]) <= 12
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
                                
                                mel_dir = os.path.join("exports", "mel_360x360", "train", folder)
                                mel_pattern = os.path.join(mel_dir, f"{segment_token}__*.png")
                                existing_mels = glob.glob(mel_pattern)

                                # Also check for stitched images (assuming they contain the segment token or are grouped by folder)
                                stitched_dir = os.path.join("exports", "mel_stitched", folder)
                                # Adjust pattern if stitched images have a different naming convention
                                stitched_pattern = os.path.join(stitched_dir, f"*{segment_token}*.png")
                                existing_stitched = glob.glob(stitched_pattern)
                                
                                if existing_mels or existing_stitched:
                                    print(
                                        f"Spectrograms already exist for {eeg_file} (360x360: {bool(existing_mels)}, Stitched: {bool(existing_stitched)})"
                                    )
                                else:
                                    print(
                                        f"No generated images found for {eeg_file}. Consider generating."
                                    )
                                    # check if the specific EEG file is already in the local folder
                                    local_file_path = f"icare_data/training/{folder}/{eeg_file}"
                                    if os.path.exists(local_file_path):
                                        print(
                                            f"EEG file {eeg_file} already exists locally at {local_file_path}"
                                        )
                                    else:
                                        print(
                                            f"EEG file {eeg_file} not found locally. Consider downloading it from {folder_path}."
                                        )
                                        # If the file doesn't exist locally, download it
                                        remote_file_path = posixpath.join(folder_path, eeg_file)
                                        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                                        print(f"Downloading {remote_file_path} to {local_file_path}...")
                                        
                                        tmp_file_path = local_file_path + ".tmp"
                                        sftp.get(remote_file_path, tmp_file_path)
                                        os.rename(tmp_file_path, local_file_path)
                                        time.sleep(1)
                        else:
                            print("  No EEG files found in this folder.")
                    except IOError as io_err:
                        print(f"  Could not list {folder_path}: {io_err}")

                    print("doing everything for folder:", folder)
                    print(folder)
                    do_everything(folder)

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

        sftp.close()
        client.close()

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

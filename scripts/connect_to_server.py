import posixpath
import paramiko
import stat
import os
from dotenv import load_dotenv

load_dotenv()
hostname = os.getenv("HOSTNAME")
port = os.getenv("PORT")
username = os.getenv("HPC_USERNAME")
private_key_path = os.getenv("PRIVATE_KEY_PATH")
remote_path = os.getenv("REMOTE_PATH")


try:
    # Initialize SSH Client
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # Load the private key
    # Use RSAKey, Ed25519Key, or the generic from_private_key_file
    key = paramiko.RSAKey.from_private_key_file(private_key_path)

    print(f"Connecting to {hostname}...")
    client.connect(hostname, port, username, pkey=key)

    # Open SFTP session to list folders
    sftp = client.open_sftp()

    print(f"Retrieving folders in: {remote_path}\n")

    # List directory and filter for directories only
    contents = sftp.listdir_attr(remote_path)
    folders = [item.filename for item in contents if stat.S_ISDIR(item.st_mode)]

    folders = ["0284", "0286"]  # Example: ['0284', '0286'] or [] to list all folders

    if folders:
        print(f"Found {len(folders)} folders:")
        print(folders)

        for folder in sorted(folders):
            folder_path = posixpath.join(remote_path, folder)
            print(f"\nListing files in {folder_path} containing 'EEG':")

            try:
                folder_contents = sftp.listdir_attr(folder_path)
                eeg_files = sorted(
                    item.filename
                    for item in folder_contents
                    if not stat.S_ISDIR(item.st_mode) and "EEG" in item.filename
                )

                if eeg_files:
                    print(eeg_files)

                    # Check if the EEG file has already a mel-spectogram
                    # mel-spectogram example: exports\mel_stitched\0284\grid_stitched_0284_001_004.png
                    # exports\mel_stitched\0286\grid_stitched_0286_002_021.png
                    for eeg_file in eeg_files:
                        mel_path = f"exports/mel_stitched/{folder}/grid_stitched_{folder}_{eeg_file.split('_')[1]}_{eeg_file.split('_')[2]}_{eeg_file.split('_')[3].split('.')[0]}.png"
                        mel_path = mel_path.replace("_EEG", "")
                        if os.path.exists(mel_path):
                            print(
                                f"Mel-spectogram already exists for {eeg_file} at {mel_path}"
                            )
                        else:
                            print(
                                f"No {mel_path} found for {eeg_file}. Consider generating it."
                            )
                else:
                    print("  No EEG files found in this folder.")
            except IOError as io_err:
                print(f"  Could not list {folder_path}: {io_err}")
    else:
        print("No folders found or path is empty.")

    sftp.close()
    client.close()

except Exception as e:
    print(f"Error: {e}")

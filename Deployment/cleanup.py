import os
import shutil

def clean_up_files_and_directories():
    """
    Removes specified files and directories, ignoring errors if they don't exist
    or are not empty (for directories).
    """
    files_to_remove = [
        "sample_pcap.zip",
        "RandomForest400IntPortCIC1718-2.pkl"
    ]

    directories_to_remove = [
        "data/",
        "sample_pcap/",
        "logs/",
        "CICFlowMeter-3.0"
    ]

    print("Starting cleanup...")

    # Remove files
    for f in files_to_remove:
        if os.path.exists(f):
            try:
                os.remove(f)
                print(f"Removed file: {f}")
            except OSError as e:
                print(f"Error removing file {f}: {e}")
        else:
            print(f"File not found, skipping: {f}")

    # Remove directories
    for d in directories_to_remove:
        if os.path.exists(d):
            try:
                # shutil.rmtree removes a directory and all its contents
                shutil.rmtree(d)
                print(f"Removed directory: {d}")
            except OSError as e:
                print(f"Error removing directory {d}: {e}")
        else:
            print(f"Directory not found, skipping: {d}")

    print("Cleanup complete.")

if __name__ == "__main__":
    clean_up_files_and_directories()
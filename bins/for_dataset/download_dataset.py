# from huggingface_hub import HfFolder
# print(HfFolder().cache_home)
# import os
# os.environ['HF_HOME'] = "/project/tts/students/yining_ws/multi_lng/F5-TTS/dataset/"
# print(os.environ.get('HF_HOME'))
# from datasets import load_dataset
# path = "DE/*.tar"
# print(dataset.cache_files)
# print(dataset) # here should only shows 90 n_shards instead of 2360
# print(next(iter(dataset['train'])))

import os
import requests
import argparse
from urllib.parse import urljoin

def parse_args():
    parser = argparse.ArgumentParser(description="Train CFM Model")
    parser.add_argument(
        "--lang", type=str, default="EN", choices=["EN", "DE", "FR", "ZH", "JA", "KO"], help="Experiment name"
    )
    parser.add_argument(
        "--index_start", type=int, default=0
    )
    parser.add_argument(
        "--index_end", type=int, default=100
    )
    return parser.parse_args()


# Function to download a file
def download_file(base_url, local_dir, filename, HF_TOKEN):
    url = urljoin(base_url, filename)
    local_path = os.path.join(local_dir, filename)
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    print(f"Downloading {filename}...")
    response = requests.get(url, headers=headers, stream=True)
    
    if response.status_code == 404:
        print(f"Error 404: {filename} not found.")
        return
    
    response.raise_for_status()  # Raise an error for other bad responses
    
    with open(local_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded {filename}")


def main():
    args = parse_args()

    # Set your Hugging Face token
    
    # os.environ.get('HF_TOKEN')
    LANG=args.lang
    # Base URL of the directory containing .tar files
    base_url = f"https://huggingface.co/datasets/amphion/Emilia-Dataset/resolve/main/{LANG}/"

    # Local directory where files are downloaded
    local_dir = f"/project/tts/students/yining_ws/multi_lng/F5-TTS/dataset/emilia/{LANG}"

    # List of expected .tar files (update this with actual filenames)
    # Create a list of filenames from 1 to 89 with leading zeros
    expected_files = [f"{LANG}-B{str(i).zfill(6)}.tar" for i in range(args.index_start, args.index_end)]
    print('expected_files: ', expected_files)
    expected_files = [expected_file for expected_file in expected_files if expected_file!="EN-B000078.tar"]

    # Get local file list
    local_files = set(os.listdir(local_dir))

    # Check which files are missing
    missing_files = set(expected_files) - local_files

    if missing_files:
        print("The following files are missing and will be downloaded:")
        for file in missing_files:
            download_file(base_url, local_dir, file, HF_TOKEN)
    else:
        print("All expected files have been downloaded successfully!")

if __name__ == "__main__":
    main()
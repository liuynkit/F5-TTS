import os
import shutil
import json
import argparse
# Set the source and destination directories
# source_dir = "/project/tts/students/yining_ws/multi_lng/F5-TTS/dataset/emilia/DE"

# -------------------------- Argument Parsing --------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(description="Train CFM Model")
    parser.add_argument(
        "--lang", type=str, default="EN", choices=["EN", "DE", "FR", "ZH", "JA", "KO"], help="Experiment name"
    )
    return parser.parse_args()

# "DE_B00000/DE_B00000_S00001/mp3/DE_B00000_S00001_W000000.mp3", 
def process_file(src_path, dest_dir, filename):
    if filename.endswith(".mp3"):
        parts = filename.split("_")
        if len(parts) >= 4:
            b_part = parts[0] + "_" + parts[1]
            s_part = b_part + "_" + parts[2]
            
            new_dir = os.path.join(dest_dir, b_part, s_part, "mp3")
            os.makedirs(new_dir, exist_ok=True)
            
            dest_file = os.path.join(new_dir, filename)
            if not os.path.exists(dest_file):
                shutil.move(src_path, dest_file)
                print(f"Moved {filename} to {dest_file}")
            else:
                print(f"File {filename} already exists in destination. Skipping.")

    if filename.endswith(".json"):
        parts = filename.split("_")
        if len(parts) >= 4:
            b_part = parts[0] + "_" + parts[1]
        
        # Specify the output JSONL file
        output_json_file = os.path.join(dest_dir, f"{b_part}.jsonl")
        # Open the output file in write mode
        with open(output_json_file, 'a', encoding='utf-8') as outfile:
            # Read each JSON file
            with open(src_path, 'r', encoding='utf-8') as infile:
                try:
                    # Load the JSON data
                    data = json.load(infile)
                    # Delete the JSON file after processing
                    os.remove(src_path)
                    print(f"Processed JSON file: {filename}")
                    # Write the JSON object as a single line in the output file
                    json.dump(data, outfile)
                    outfile.write('\n')
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in file: {filename}")

def main():
    args = parse_args()

    # source_dir = "/project/tts/students/yining_ws/multi_lng/F5-TTS/bins/for_dataset/SOURCE_DIR"
    LANG=args.lang
    source_dir=f"/project/tts/students/yining_ws/multi_lng/F5-TTS/emilia/{LANG}"
    dest_dir = source_dir

    # Iterate over next-level subdirectories
    for b_part in os.listdir(source_dir):
        b_part_path = os.path.join(source_dir, b_part)
        if os.path.isdir(b_part_path):
            print(f"Processing directory started: {b_part}")
        # Specify the output JSONL file
        # output_json_file = os.path.join(dest_dir, f"{b_part}.jsonl")
        # Open the output file in write mode
        # with open(output_json_file, 'w', encoding='utf-8') as outfile:
        # Process files in this b_part subdirectory
            for filename in os.listdir(b_part_path):
                src_path = os.path.join(b_part_path, filename)
                if os.path.isfile(src_path):
                    process_file(src_path, dest_dir, filename)
            print(f"Processing directory Finished: {b_part}")

    print("File organization complete.")


if __name__ == "__main__":
    main()
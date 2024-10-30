#!/bin/bash
export PATH="/home/yliu/miniconda3/envs/e2tts/bin:$PATH"
MY_PATH="/project/tts/students/yining_ws/multi_lng/F5-TTS/bins/for_dataset"
cd $MY_PATH
LANG="ZH"

# Default to running all parts
RUN_PART_1=true
RUN_PART_2=true
RUN_PART_3=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --part1) RUN_PART_1=true; RUN_PART_2=false; RUN_PART_3=false ;;
        --part2) RUN_PART_1=false; RUN_PART_2=true; RUN_PART_3=false ;;
        --part3) RUN_PART_1=false; RUN_PART_2=false; RUN_PART_3=true ;;
        --part1-2) RUN_PART_1=true; RUN_PART_2=true; RUN_PART_3=false ;;
        --part2-3) RUN_PART_1=false; RUN_PART_2=true; RUN_PART_3=true ;;
        # Add more combinations as needed
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

# Rest of your script...
if $RUN_PART_1; then
    echo "Running Part 1: Download dataset"
    python download_dataset.py --lang $LANG --index_start 0 --index_end 100
fi

if $RUN_PART_2; then
    echo "Running Part 2: Extract tar files"
    # Your tar extraction code here
    # Directory containing the .tar files
    SOURCE_DIR="/project/tts/students/yining_ws/multi_lng/F5-TTS/dataset/emilia/$LANG"
    # Directory where you want to extract the files
    EXTRACT_DIR="/project/tts/students/yining_ws/multi_lng/F5-TTS/emilia/$LANG"
    # Iterate over all .tar files in the source directory
    for tarfile in "$SOURCE_DIR"/$LANG*.tar; do
        if [ -f "$tarfile" ]; then
            # Extract the basename of the tar file (without path)
            filename=$(basename "$tarfile")
            # Create a subdirectory for each tar file (optional)
            mkdir -p "$EXTRACT_DIR/${filename%.tar}"
            # Extract the contents of the tar file
            tar -xvf "$tarfile" -C "$EXTRACT_DIR/${filename%.tar}"
            echo "Extracted: $filename"
        fi
    done
    echo "Extraction complete."
fi

if $RUN_PART_3; then
    echo "Running Part 3: Rebuild directory and JSONL"
    python rebuild_dir_jsonl.py --lang $LANG
fi
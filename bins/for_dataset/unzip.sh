#!/bin/bash
LANG="FR"
# Directory containing the .tar files
SOURCE_DIR="/project/tts/students/yining_ws/multi_lng/F5-TTS/dataset/emilia/$LANG"
# SOURCE_DIR="/project/tts/students/yining_ws/multi_lng/F5-TTS/dataset/test_dataset/DE"

# Directory where you want to extract the files
EXTRACT_DIR="/project/tts/students/yining_ws/multi_lng/F5-TTS/emilia/$LANG"
# "/project/tts/students/yining_ws/multi_lng/F5-TTS/dataset/emilia/DE"

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
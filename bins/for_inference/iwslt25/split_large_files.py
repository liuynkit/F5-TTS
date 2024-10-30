def split_file_custom(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Split lines into the specified ranges
    ranges = [
        (0, 15000),      # First file: lines 1 to 15000
        (15000, 30000),  # Second file: lines 15001 to 30000
        (30000, 45000),  # Third file: lines 30001 to 45000
        (45000, 60000),  # Fourth file: lines 45001 to 60000
    ]
    
    # Write the split files
    for i, (start, end) in enumerate(ranges, 1):
        with open(f'{filename}.part{i}', 'w') as f_part:
            f_part.writelines(lines[start:end])

# Example usage
split_file_custom('/project/tts/students/yining_ws/multi_lng/F5-TTS/inputs/iwslt25/60000_with_info.txt')

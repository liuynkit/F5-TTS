input_file = "/project/tts/students/yining_ws/multi_lng/F5-TTS/outputs/iwslt25_augmented_60000/augmented_all_filelist.csv"
results = []
with open(input_file, 'r', encoding='utf-8') as file:
    for idx, line in enumerate(file):
        print("idx: ", idx)
        if idx==0:
            continue       
        # Strip whitespace and split by '|'
        split_line = [item.strip().strip('\n') for item in line.strip().split('|')]
        if len(split_line) == 5:  # Only add non-empty lines
            gen_audio, gen_text, en_tran, ref_audio, ref_text = split_line
            results.append(split_line)

print(len(results))
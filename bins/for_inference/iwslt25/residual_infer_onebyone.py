results = []
input_file = "/project/tts/students/yining_ws/multi_lng/F5-TTS/outputs/iwslt25_augmented_60000/augmented_all_filelist.csv"
with open(input_file, 'r', encoding='utf-8') as file:
    for idx, line in enumerate(file):
        print("idx: ", idx)
        if idx==0:
            continue       
        # Strip whitespace and split by '|'
        split_line = [item.strip().strip('\n') for item in line.strip().split('|')]
        if len(split_line) == 5:  # Only add non-empty lines
            # gen_audio, 
            gen_wav, gen_text, en_tran, ref_audio, ref_text = split_line
            items = []
            
            items ="|".join([f"augmented_{idx}.wav", gen_text, en_tran, ref_audio, ref_text])
            results.append(items)

output_txt_file = "/project/tts/students/yining_ws/multi_lng/F5-TTS/outputs/iwslt25_augmented_withbatch_60000.csv"
with open(output_txt_file, 'w', encoding='utf-8') as txtfile:
    txtfile.write("gen_audio|gen_text|gen_text_en_tran|ref_audio|ref_text" + '\n')
    for result in results:
        txtfile.write(result + '\n')
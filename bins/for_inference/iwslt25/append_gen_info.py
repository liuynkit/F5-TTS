import csv
import random


input_file = "/project/tts/students/yining_ws/multi_lng/F5-TTS/inputs/iwslt25/60000_with_info.txt"
# ref_info_file = "/project/OML/zli/iwslt2025/data/bem/training/bigc/bigc/data/bem/hfdata/processed_train.csv"
output_txt_file = "/project/tts/students/yining_ws/multi_lng/F5-TTS/outputs/iwslt25_augmented_60000/augmented_filelist.csv"

results = []
with open(input_file, 'r', encoding='utf-8') as file:
    for idx, line in enumerate(file):
                
        # Strip whitespace and split by '|'
        split_line = [item.strip().strip('\n') for item in line.strip().split('|')]
        if len(split_line) == 3:  # Only add non-empty lines
            gen_text, ref_audio, ref_text = split_line
            items = []

            gen_audio = "augmented_"+str(idx+1)+".wav"
            # self.infer_one(ref_audio, ref_text, gen_text, idx+start_index, wav_save_dir)
            items.append(gen_audio)
            items +=split_line

            results.append("|".join(items))
            # progress_bar.update(1)
            # progress_bar.set_postfix(update=str(global_update), loss=loss.item())
# Write sampled sentences to a txt file
with open(output_txt_file, 'w', encoding='utf-8') as txtfile:
    txtfile.write("gen_audio|gen_text|ref_audio|ref_text" + '\n')
    for result in results:
        txtfile.write(result + '\n')






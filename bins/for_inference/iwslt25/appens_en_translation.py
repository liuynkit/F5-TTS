import csv
import random
import difflib
import re

def fuzzy_match(gen_text, dict_tran, threshold=0.9):
    words = gen_text.split(" ")
    words = re.findall(r'\b[a-zA-Z]+\b', gen_text)
    print('debug:', words[0])

    for key in dict_tran:
        if words[0] not in key:
            continue
        similarity = difflib.SequenceMatcher(None, gen_text, key).ratio()
        if similarity >= threshold:
            print(f"gen_text '{gen_text}' is similar to key '{key}' with {similarity:.2%} similarity.")
            return True, key
    return False, None



input_file = "/project/tts/students/yining_ws/multi_lng/F5-TTS/outputs/iwslt25_augmented_60000/augmented_filelist.csv"
# ref_info_file = "/project/OML/zli/iwslt2025/data/bem/training/bigc/bigc/data/bem/hfdata/processed_train.csv"
output_txt_file = "/project/tts/students/yining_ws/multi_lng/F5-TTS/outputs/iwslt25_augmented_60000/augmented_all_filelist.csv"

en_input_file = "/project/OML/zli/iwslt2025/data/bem/training/mt_data_all.csv"

dict_tran = {}
with open(en_input_file, 'r', encoding='utf-8') as file:
    for idx, line in enumerate(file):
        if idx==0:
            continue       
        # Strip whitespace and split by '|'
        split_line = [item.strip().strip('\n') for item in line.strip().split('|')]
        if len(split_line) == 3:  # Only add non-empty lines
            en, bem, source = split_line
            if source!='nllb':
                continue
            dict_tran[bem] = en

results = []
with open(input_file, 'r', encoding='utf-8') as file:
    for idx, line in enumerate(file):
        print("idx: ", idx)
        if idx==0:
            continue       
        # Strip whitespace and split by '|'
        split_line = [item.strip().strip('\n') for item in line.strip().split('|')]
        if len(split_line) == 4:  # Only add non-empty lines
            gen_audio, gen_text, ref_audio, ref_text = split_line
            items = []

            if gen_text not in dict_tran:
                flag, key = fuzzy_match(gen_text, dict_tran)
                if not flag:
                    print('gen_text: ', gen_text)
                    assert 0==1
            else:
                key = gen_text
            en_tran = dict_tran[key]
            items.append(gen_audio)
            items.append(gen_text)
            items.append(en_tran)
            # self.infer_one(ref_audio, ref_text, gen_text, idx+start_index, wav_save_dir)
            items.append(ref_audio)
            items.append(ref_text)
            results.append("|".join(items))
            # progress_bar.update(1)
            # progress_bar.set_postfix(update=str(global_update), loss=loss.item())
# Write sampled sentences to a txt file
with open(output_txt_file, 'w', encoding='utf-8') as txtfile:
    txtfile.write("gen_audio|gen_text|gen_text_en_tran|ref_audio|ref_text" + '\n')
    for result in results:
        txtfile.write(result + '\n')






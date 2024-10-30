import csv
import random
def load_gen_data(file_path):
    results = []
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for idx, line in enumerate(lines):
            # Strip whitespace and split by '|'
            split_line = [item.strip() for item in line.strip().split('|')]
            # if len(split_line) == 1:  # Only add non-empty lines
            gen_text = split_line[0].strip("\n")
            results.append(gen_text)
    return results
        # Write sampled sentences to a txt file
        # with open(output_file, 'w', encoding='utf-8') as txtfile:
        #     for result in results:
        #         txtfile.write(result + '\n')


def load_ref_data(file_path, sample_size=60000, output_file=None, vocab_list=None):
    data = []
    char_list = ['#', ')', "(", "*", "@", "-", "🙂", "✂️", "+", "0", "🎵", "▶", ">", "[", "]", \
        "1","2","3","4","5","6","7","8","9", "🐒", "🤑", "😂", "=", "😀", "😋", "🇿🇲", "🔥", "_", "மொழி"]
    
    # num = 0
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter='|')
        for row in csv_reader:
            if len(row) == 4:
                audio, bemba, eng, duration = row
                # bemba = remove_emoji(bemba)
                if duration =="duration":
                    continue
                # print(row)
                if float(duration) > 10.0 or float(duration)< 3.0:
                    continue
                # if contains_emoji(bemba):
                #     continue
                # # print(get_out_of_vocab_chars(bemba, vocab_list))
                # if has_out_of_vocab_chars(bemba, vocab_list):
                #     continue
                # if len(bemba)>=200 or len(bemba)<18:
                #     continue
                # if contains_chars(bemba, char_list):
                #     continue
                data.append({
                    'audio': audio,
                    'bemba': bemba,
                    # 'source': data_source
                })
                # num+=1

    # Randomly sample sentences
    if len(data) > sample_size:
        sampled_data = random.sample(data, sample_size)
    else:
        print(f"ALERT: data {len(data)} not enought for {sample_size} !!!")
        sampled_data = data
    # # Write sampled sentences to a txt file
    # with open(output_file, 'w', encoding='utf-8') as txtfile:
    #     for sentence in sampled_data:
    #         txtfile.write(sentence['bemba'] + '\n')

    # print(f"Sampled {len(sampled_data)} sentences and wrote them to {output_file}")
    return sampled_data

input_file = "/project/tts/students/yining_ws/multi_lng/F5-TTS/inputs/iwslt25/mt_data_forinfer_10.txt"
ref_info_file = "/project/OML/zli/iwslt2025/data/bem/training/bigc/bigc/data/bem/hfdata/processed_train.csv"

sampled_data_ref = load_ref_data(ref_info_file, sample_size=10)
sampled_data_gen = load_gen_data(input_file)

zipped = zip(sampled_data_gen, sampled_data_ref)
print(f"Gen data count: {len(sampled_data_gen)}, Ref data count: {len(sampled_data_ref)}")

count_gen = sum(1 for item in sampled_data_gen if "\n" in item)
count_ref_audio = sum(1 for item in sampled_data_ref if "\n" in item['audio'])
count_ref_bemba = sum(1 for item in sampled_data_ref if "\n" in item['bemba'])
for item in sampled_data_ref:
    if "\n" in item['bemba']:
        # print(item['bemba'])
        print(item['bemba'].replace("\n", " ").strip())
        print("######\n")
print(f"Newlines in data_gen: {count_gen}")
print(f"Newlines in ref_audio: {count_ref_audio}")
print(f"Newlines in ref_bemba: {count_ref_bemba}")

output_file = "/project/tts/students/yining_ws/multi_lng/F5-TTS/inputs/iwslt25/10_with_info.txt"
# Write sampled sentences to a txt file
with open(output_file, 'w', encoding='utf-8') as txtfile:
    # Iterate over the zipped object
    for idx, (data_gen, data_ref) in enumerate(zipped):
        # print("idx: ", idx)
        items = []
        items.append(data_gen.strip('\n'))
        items.append(data_ref['audio'].strip('\n'))
        items.append(data_ref['bemba'].replace("\n", " ").strip())
        # Replace newlines with spaces and remove leading/trailing whitespace
        single_line_item = "|".join(items).strip()
        txtfile.write(f"{single_line_item}\n")
        # line = data_gen+"|"+data_ref['audio']+"|"+data_ref['bemba']
        # txtfile.write(line + '\n')
    # for result in results:
    #     txtfile.write(result + '\n')

# results = []
# file_path = "/project/tts/students/yining_ws/multi_lng/F5-TTS/inputs/iwslt25/60000_with_info.txt"
# with open(file_path, 'r', encoding='utf-8') as file:
#     for idx, line in enumerate(file):
#         # Strip whitespace and split by '|'
#         split_line = [item.strip() for item in line.strip().split('|')]
#         if len(split_line) == 3:  # Only add non-empty lines
#             # gen_text = split_line[0]
#             results.append(split_line)
# print(len(results))

    # print(f"Number: {num}, Letter: {letter}")





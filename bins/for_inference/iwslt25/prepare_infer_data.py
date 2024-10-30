import csv
import random
import re
from pathlib import Path
import fasttext

# https://fasttext.cc/docs/en/language-identification.html
# model_file = Path(__file__).parent / 'lid.176.bin'
# if not model_file.is_file():
#     raise FileNotFoundError('Run wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin in'
#                             ' ' + str(model_file.parent))
lang_id_model = fasttext.load_model("/project/tts/students/yining_ws/multi_lng/F5-TTS/ckpts/fasttext/lid.176.bin")

text = "This is very good."
(language,), prob = lang_id_model.predict(text)
language = language[len('__label__'):]
print(language)
print(prob)

def test_language(string):
    (language,), prob = lang_id_model.predict(text)
    language = language[len('__label__'):]
    # print("text: ", string)
    # print("prob: ", prob)
    # print("language: ", language)
    if language=='en' and prob>0.95:
        return True
    return False

def remove_emoji(string):
    # Define the emoji pattern
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    
    # Remove emojis
    string_without_emojis = emoji_pattern.sub(r'', string)
    
    # Remove extra spaces left behind
    cleaned_string = re.sub(r'\s+', ' ', string_without_emojis).strip()
    
    return cleaned_string

def has_out_of_vocab_chars(string, vocab):
    return any(char not in vocab for char in string)

def get_out_of_vocab_chars(string, vocab):
    return [char for char in string if char not in vocab]

def contains_chars(string, char_list):
    return any(char in string for char in char_list)

def contains_emoji(string):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    return bool(emoji_pattern.search(string))

def load_translation_data(file_path, sample_size=60000, output_file=None, vocab_list=None):
    data = []
    char_list = ['#', ')', "(", "*", "@", "-", "🙂", "✂️", "+", "0", "🎵", "▶", ">", "[", "]", \
        "1","2","3","4","5","6","7","8","9", "🐒", "🤑", "😂", "=", "😀", "😋", "🇿🇲", "🔥", "_", "மொழி"]

    
    # num = 0
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter='|')
        for row in csv_reader:
            if len(row) == 3:
                english, bemba, data_source = row
                if data_source!='nllb':
                    continue
                # bemba = remove_emoji(bemba)
                if contains_emoji(bemba):
                    continue
                # print(get_out_of_vocab_chars(bemba, vocab_list))
                if has_out_of_vocab_chars(bemba, vocab_list):
                    continue
                if len(bemba)>=200 or len(bemba)<18:
                    continue
                if contains_chars(bemba, char_list):
                    continue
                # if test_language(bemba):
                #     continue
                data.append({
                    # 'english': english,
                    'bemba': bemba,
                    # 'source': data_source
                })
                # num+=1

    # Randomly sample sentences
    if len(data) > sample_size:
        sampled_data = random.sample(data, sample_size)
    else:
        print(f"ALERT: data not enought for {sample_size} !!!")
        sampled_data = data

    # Write sampled sentences to a txt file
    with open(output_file, 'w', encoding='utf-8') as txtfile:
        for sentence in sampled_data:
            txtfile.write(sentence['bemba'] + '\n')

    print(f"Sampled {len(sampled_data)} sentences and wrote them to {output_file}")
    return sampled_data

def read_vocab(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f)


file_path = '/project/OML/zli/iwslt2025/data/bem/training/mt_data_all.csv'
output_path = "/project/tts/students/yining_ws/multi_lng/F5-TTS/inputs/iwslt25/mt_data_forinfer_10.txt"


tocheck_vocab_path = "/project/tts/students/yining_ws/multi_lng/F5-TTS/preprocessed_data/bigc_bem_char/vocab.txt"
vocab_tocheck = read_vocab(tocheck_vocab_path)
vocab_tocheck.add(" ")

translation_data = load_translation_data(file_path, sample_size=10, output_file=output_path, vocab_list=vocab_tocheck)

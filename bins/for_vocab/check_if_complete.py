refer_vocab_path = "/project/tts/students/yining_ws/multi_lng/F5-TTS/data/Emilia_ZH_EN_pinyin/vocab.txt"
tocheck_vocab_path = "/project/tts/students/yining_ws/multi_lng/F5-TTS/preprocessed_data/bigc_bem_char/vocab.txt"

def read_vocab(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f)

# Read both vocab files
vocab_ref = read_vocab(refer_vocab_path)
vocab_tocheck = read_vocab(tocheck_vocab_path)

# Check if all characters in A are included in B
missing_chars = vocab_tocheck - vocab_ref

if " " in vocab_tocheck:
    print("space in the list!")
else:
    print("space not in the list!")

if not missing_chars:
    print("All characters in vocab A are included in vocab B.")
else:
    print("The following characters from vocab A are missing in vocab B:")
    for char in missing_chars:
        print(char)

print(f"Total missing characters: {len(missing_chars)}")

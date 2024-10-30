def read_vocab(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f)


data_dir = "/project/tts/students/yining_ws/multi_lng/F5-TTS"


langs1 = ["FR"]
tokenizer = "pinyin"

langs2 = ["ZH", "EN"]
dataset_name1 = f"Emilia_{'_'.join(langs1)}_{tokenizer}"
dataset_name2 = f"Emilia_{'_'.join(langs2)}_{tokenizer}"
# Read both vocab files
vocab_a = read_vocab(f"{data_dir}/data/{dataset_name1}/vocab.txt")
vocab_b = read_vocab(f"{data_dir}/data/{dataset_name2}/vocab.txt")

# Check if all characters in A are included in B
missing_chars = vocab_a - vocab_b

if not missing_chars:
    print("All characters in vocab A are included in vocab B.")
else:
    print("The following characters from vocab A are missing in vocab B:")
    for char in missing_chars:
        print(char)

print(f"Total missing characters: {len(missing_chars)}")
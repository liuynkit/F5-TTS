def clean_the_vocab(tokenizer_path, tokenizer_path1):
    with open(tokenizer_path, "r", encoding="utf-8") as f:
        vocab_char_map = {}
        vocab_char_map[" "] = 0
        idx = 1
        for i, char in enumerate(f):
            # print(f"#{char[:-1]}#")
            if char[:-1]!=" " and char[:-1] not in list(vocab_char_map.keys()):
                vocab_char_map[char[:-1]] = idx
                idx+=1

        
    print('debug: ', vocab_char_map)
    # Write each character to a new line in the file
    with open(tokenizer_path, "w") as f:
        for vocab in list(vocab_char_map.keys()):
            f.write(vocab + "\n")  # Ensure each character is on a new line

if __name__ == "__main__":
    tokenizer_path = "/project/tts/students/yining_ws/multi_lng/F5-TTS/preprocessed_data/bigc_bem_char/vocab.txt"
    tokenizer_path1 = "/project/tts/students/yining_ws/multi_lng/F5-TTS/preprocessed_data/bigc_bem_char/vocab1.txt"
    clean_the_vocab(tokenizer_path, tokenizer_path1)
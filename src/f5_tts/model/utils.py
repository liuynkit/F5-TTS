from __future__ import annotations

import os
import random
from collections import defaultdict
from importlib.resources import files

import torch
from torch.nn.utils.rnn import pad_sequence

import jieba
from pypinyin import lazy_pinyin, Style

import sys
path_to_add = '/project/tts/students/yining_ws/multi_lng/F5-TTS/src'
# Check if the path is already in sys.path
if path_to_add not in sys.path:
    sys.path.append(path_to_add)
    print(f"Added {path_to_add} to sys.path")
else:
    print(f"{path_to_add} is already in sys.path")


# custion pad_sequence for onnx export
def custom_pad_sequence(sequences, padding_value=0, batch_first=True):
    max_len = max([s.size(0) for s in sequences])
    out_tensors = []
    for tensor in sequences:
        padding_size = max_len - tensor.size(0)
        padded = torch.nn.functional.pad(tensor, (0, 0, 0, padding_size), value=padding_value)
        out_tensors.append(padded)
    return torch.stack(out_tensors, dim=0)


# seed everything


def seed_everything(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# helpers


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


# tensor helpers


def lens_to_mask(t: int["b"], length: int | None = None) -> bool["b n"]:  # noqa: F722 F821
    if not exists(length):
        length = t.amax()

    seq = torch.arange(length, device=t.device)
    return seq[None, :] < t[:, None]


def mask_from_start_end_indices(seq_len: int["b"], start: int["b"], end: int["b"]):  # noqa: F722 F821
    max_seq_len = seq_len.max().item()
    seq = torch.arange(max_seq_len, device=start.device).long()
    start_mask = seq[None, :] >= start[:, None]
    end_mask = seq[None, :] < end[:, None]
    return start_mask & end_mask


def mask_from_frac_lengths(seq_len: int["b"], frac_lengths: float["b"]):  # noqa: F722 F821
    lengths = (frac_lengths * seq_len).long()
    max_start = seq_len - lengths

    rand = torch.rand_like(frac_lengths)
    start = (max_start * rand).long().clamp(min=0)
    end = start + lengths

    return mask_from_start_end_indices(seq_len, start, end)


def maybe_masked_mean(t: float["b n d"], mask: bool["b n"] = None) -> float["b d"]:  # noqa: F722
    if not exists(mask):
        return t.mean(dim=1)

    t = torch.where(mask[:, :, None], t, torch.tensor(0.0, device=t.device))
    num = t.sum(dim=1)
    den = mask.float().sum(dim=1)

    return num / den.clamp(min=1.0)


# simple utf-8 tokenizer, since paper went character based
def list_str_to_tensor(text: list[str], padding_value=-1) -> int["b nt"]:  # noqa: F722
    list_tensors = [torch.tensor([*bytes(t, "UTF-8")]) for t in text]  # ByT5 style
    text = pad_sequence(list_tensors, padding_value=padding_value, batch_first=True)
    return text


# char tokenizer, based on custom dataset's extracted .txt file
def list_str_to_idx(
    text: list[str] | list[list[str]],
    vocab_char_map: dict[str, int],  # {char: idx}
    padding_value=-1,
) -> int["b nt"]:  # noqa: F722
    list_idx_tensors = [torch.tensor([vocab_char_map.get(c, 0) for c in t]) for t in text]  # pinyin or char style
    text = pad_sequence(list_idx_tensors, padding_value=padding_value, batch_first=True)
    return text


# Get tokenizer


def get_tokenizer(dataset_name, tokenizer: str = "pinyin"):
    """
    tokenizer   - "pinyin" do g2p for only chinese characters, need .txt vocab_file
                - "char" for char-wise tokenizer, need .txt vocab_file
                - "byte" for utf-8 tokenizer
                - "custom" if you're directly passing in a path to the vocab.txt you want to use
    vocab_size  - if use "pinyin", all available pinyin types, common alphabets (also those with accent) and symbols
                - if use "char", derived from unfiltered character & symbol counts of custom dataset
                - if use "byte", set to 256 (unicode byte range)
    """
    if tokenizer in ["pinyin", "char"]:
        tokenizer_path = os.path.join(files("f5_tts").joinpath("../../data"), f"{dataset_name}_{tokenizer}/vocab.txt")
    
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            vocab_char_map = {}
            for i, char in enumerate(f):
                # print(f"#{char}#")
                # to remove the line changing icon
                # print(f"#{char[:-1]}#")
                vocab_char_map[char[:-1]] = i
        vocab_size = len(vocab_char_map)
        assert vocab_char_map[" "] == 0, "make sure space is of idx 0 in vocab.txt, cuz 0 is used for unknown char"

    elif tokenizer == "byte":
        vocab_char_map = None
        vocab_size = 256

    elif tokenizer == "custom":
        with open(dataset_name, "r", encoding="utf-8") as f:
            vocab_char_map = {}
            for i, char in enumerate(f):
                vocab_char_map[char[:-1]] = i
        vocab_size = len(vocab_char_map)

    return vocab_char_map, vocab_size



def clean_the_vocab(tokenizer_path):
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



# convert char to pinyin

def convert_char_to_pinyin(text_list, polyphone=True):
    final_text_list = []
    god_knows_why_en_testset_contains_zh_quote = str.maketrans(
        {"“": '"', "”": '"', "‘": "'", "’": "'"}
    )  # in case librispeech (orig no-pc) test-clean
    custom_trans = str.maketrans({";": ","})  # add custom trans here, to address oov
    for text in text_list:
        char_list = []
        text = text.translate(god_knows_why_en_testset_contains_zh_quote)
        text = text.translate(custom_trans)
        for seg in jieba.cut(text):
            seg_byte_len = len(bytes(seg, "UTF-8"))
            if seg_byte_len == len(seg):  # if pure alphabets and symbols
                if char_list and seg_byte_len > 1 and char_list[-1] not in " :'\"":
                    char_list.append(" ")
                char_list.extend(seg)
            elif polyphone and seg_byte_len == 3 * len(seg):  # if pure chinese characters
                seg = lazy_pinyin(seg, style=Style.TONE3, tone_sandhi=True)
                for c in seg:
                    if c not in "。，、；：？！《》【】—…":
                        char_list.append(" ")
                    char_list.append(c)
            else:  # if mixed chinese characters, alphabets and symbols
                for c in seg:
                    if ord(c) < 256:
                        char_list.extend(c)
                    else:
                        if c not in "。，、；：？！《》【】—…":
                            char_list.append(" ")
                            char_list.extend(lazy_pinyin(c, style=Style.TONE3, tone_sandhi=True))
                        else:  # if is zh punc
                            char_list.append(c)
        final_text_list.append(char_list)

    return final_text_list


# filter func for dirty data with many repetitions


def repetition_found(text, length=2, tolerance=10):
    pattern_count = defaultdict(int)
    for i in range(len(text) - length + 1):
        pattern = text[i : i + length]
        pattern_count[pattern] += 1
    for pattern, count in pattern_count.items():
        if count > tolerance:
            return True
    return False


def main():
    text_list = ["This 15 sample text contains laughing \n emojis 😀 😃 😄 😁 😆 😅 😂 🤣", "I love bitte lemon.", "我很喜欢猫。"]
    final_text_list = convert_char_to_pinyin(text_list)
    print('final_text_list: ', final_text_list)
    dataset_name = "bigc_bem"
    get_tokenizer(dataset_name, tokenizer="char")


if __name__ == "__main__":
    # Code to run when the script is executed directly
    main()
import torch


def is_end(text):
    text = text.strip()
    if text[-1]=="." or 
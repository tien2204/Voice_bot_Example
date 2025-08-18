from tqdm.notebook import tqdm
from alignment_utils import *


from phonemizer import phonemize
import phonemizer

path = "metadata.csv"

with open(path, "r") as f:
    all_text = f.readlines()
all_text = [i.strip() for i in all_text]

path_phoneme_out = "phonemes_metadata.jsonl"

import random

random.shuffle(all_text)

import jsonlines

phoneme_text = []
metadata_cleaned = []
with jsonlines.open(path_phoneme_out, "w") as fphoneme:
    for text in tqdm(all_text):
        try:
            content = text.split("|")[-1].strip()
            phoneme = phonemize(str(content), language="vi").strip().split(" ")
            if phoneme[-1] == "":
                phoneme = phoneme[:-1]
            if phoneme[-1] == ".":
                phoneme = phoneme[:-1]
            elif phoneme[-1] == "..":
                phoneme[-1] = "."
            if len(phoneme) < 2:
                continue
            fphoneme.write(
                {"text": content, "phonemes": phoneme, "audio": text.split("|")[0]}
            )
        except:
            continue

print("Done.")

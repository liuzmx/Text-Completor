import json
import random
import uuid
from typing import List

from nltk import sent_tokenize, word_tokenize


def mask_phrase(sentences: List[str], mask_token: str = "[MASK]"):

    for index, sentence in enumerate(sentences):
        input_text = "".join(sentences[:index]) + " " + mask_token + " " + "".join(sentences[index + 1 :])
        yield input_text.strip() + "\t" + sentence.strip()


def process_content(content: str, sentence_nums: int = 3):

    sentences = sent_tokenize(content)
    if len(sentences) <= sentence_nums:
        return mask_phrase(sentences)

    for i in range(0, len(sentences) - sentence_nums, sentence_nums):
        return mask_phrase(sentences[i : i + sentence_nums])


def generate_dataset(input_file, output_file, sentence_nums: int = 3):
    with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        for line in infile:
            data = json.loads(line)
            content = data.get("title", "") + ". " + data.get("content", "")
            content = content.replace("\n", " ")
            samples = []
            for sample in process_content(content, sentence_nums):
                samples.append(sample + "\n")
            outfile.writelines(list(set(samples)))


# 使用示例
generate_dataset("../data/bbc_news/bbc_news_2024-11.jsonl", "train.txt", sentence_nums=3)
# shuf -n 10000 train.txt > train_10k.txt

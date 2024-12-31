import json
import os
import random

from nltk import sent_tokenize


def build_dataset(input_dir: str, output_path: str, min_length=2, max_length=8):

    for name in os.listdir(input_dir):
        if not name.endswith(".jsonl"):
            continue

    data = []
    with open(os.path.join(input_dir, name), "r") as f:
        for line in f.readlines():
            item = json.loads(line)
            sentences = sent_tokenize(item["content"])
            for sentence in sentences:
                words = sentence.split()
                if len(words) < max_length:
                    continue
                start = random.randint(0, len(words) - max_length)
                end = start + random.randint(min_length, max_length)
                data.append(
                    json.dumps(
                        {
                            "original_text": sentence,
                            "input_text": " ".join(words[:start] + ["[MASK]"] + words[end:]),
                            "target_text": " ".join(words[start:end]),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    with open(output_path, "w+") as f:
        f.writelines(data)


if __name__ == "__main__":
    # build_dataset("../data/bbc_news/2024-09", "test_with_knowledge_history.jsonl")
    build_dataset("../data/bbc_news/2024-08", "test_without_knowledge_history.jsonl")

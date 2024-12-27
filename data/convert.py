import json
import os


def convert(jsonl_dir: str, txt_dir: str) -> None:
    """
    将jsonl文件转换为txt文件
    """
    if not os.path.exists(txt_dir):
        os.mkdir(txt_dir)
    for name in os.listdir(jsonl_dir):
        if not name.endswith(".jsonl"):
            continue
        paragraphs = []
        with open(os.path.join(jsonl_dir, name), "r") as f:
            for line in f.readlines():
                item = json.loads(line)
                paragraphs.append(f"{item['title']}\n{item['content']}")
        with open(os.path.join(txt_dir, name.replace(".jsonl", ".txt")), "w+") as f:
            f.write("\n\n".join(paragraphs))


if __name__ == "__main__":
    convert("./bbc_news", "./bbc_news_txt")

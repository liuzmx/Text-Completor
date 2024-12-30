import json
import logging
import os

from trie import TrieClient

logger = logging.getLogger(__name__)


def build_bbc_news_trie(news_path: str, pkl_path: str):
    """
    构建BBC新闻Trie树
    """
    client = TrieClient()
    for name in os.listdir(news_path):
        if not name.endswith(".jsonl"):
            continue
        file_path = os.path.join(news_path, name)
        with open(file_path, "r") as f:
            for line in f.readlines():
                item = json.loads(line)
                client.trie.insert(f"{item['title']}\n{item['content']}".lower())
    client.save(pkl_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_bbc_news_trie("../data/bbc_news/example", "./trie/pkls/trie_example.pkl")

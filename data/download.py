import shutil
from typing import List

from datasets import load_dataset


def download_bbc_news(tags: List[str], file_dir: str = "./bbc_news"):
    """
    下载HF上的BBC新闻数据集
    param tags: List[str], 标签列表，如：["2024-11", "2024-10"]
    param file_dir: str, 文件保存路径
    """
    cache_dir = ".cache"
    for tag in tags:
        ds = load_dataset("RealTimeData/bbc_news_alltime", tag, cache_dir=cache_dir)
        ds["train"].to_json(f"{file_dir}/bbc_news_{tag}.jsonl")
    shutil.rmtree(cache_dir)


if __name__ == "__main__":
    download_bbc_news(["2024-11", "2024-10", "2024-09"])

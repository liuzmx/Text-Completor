import logging

from knowledge import LocalKnowledgeBase

logger = logging.getLogger(__name__)


def build_bbc_news_local_knowledge(input_dir: str) -> None:
    """
    构建BBC新闻本地知识库
    """
    client = LocalKnowledgeBase()

    client.create_collection()
    docs = client.load_and_split(input_dir)
    client.insert(docs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    input_dir = "../data/bbc_news/2024-09"
    build_bbc_news_local_knowledge(input_dir)

from typing import List

from common import CONFIG
from openai import OpenAI


def embed_texts(input: List[str], conf: dict = CONFIG["openai"]) -> List[List[float]]:
    """
    使用Embedding API对文本进行嵌入
    :param docs: 文本列表
    :return: 嵌入列表
    """

    client = OpenAI(api_key=conf["api_key"], base_url=conf["base_url"])
    response = client.embeddings.create(
        input=input, model=conf["embedding"]["model"], dimensions=conf["embedding"]["dimensions"]
    )
    vectors = [item.embedding for item in response.data]
    return vectors

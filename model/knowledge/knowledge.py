import json
import logging
import os
import uuid
from datetime import datetime
from typing import List, Optional

from common import CONFIG
from llama_index.core.node_parser import SentenceSplitter
from pydantic import BaseModel
from pymilvus import MilvusClient

from .embedding import embed_texts

logger = logging.getLogger(__name__)


class KnowledgeDoc(BaseModel):
    """
    知识库文档
    """

    text: str
    metadata: dict = {}
    uuid: Optional[str] = None
    vector: Optional[List[float]] = None


class LocalKnowledgeBase:
    """
    用户本地知识库
    """

    def __init__(self, conf: dict = CONFIG) -> None:
        self.conf = conf
        self.client = MilvusClient(f"{conf['knowledge']['dir']}/{conf['knowledge']['db']}")
        self.collection_name = conf["knowledge"]["collection"]

    def create_collection(self, remove_if_exists: bool = False) -> None:
        """
        创建集合
        """
        if self.client.has_collection(collection_name=self.collection_name):
            logger.info(f"collection {self.collection_name} exists")
            if remove_if_exists:
                logger.info(f"remove collection {self.collection_name}")
                self.client.drop_collection(collection_name=self.collection_name)
        else:
            logger.info(f"create collection {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                dimension=self.conf["openai"]["embedding"]["dimensions"],
                auto_id=True,
            )

    def insert(self, docs: List[KnowledgeDoc], chunk_size: int = 32) -> None:
        """
        插入数据
        param texts: 文本列表
        """

        for skip in range(0, len(docs), chunk_size):
            chunk = docs[skip : skip + chunk_size]
            logger.info(f"insert {skip} to {skip + chunk_size}, total {len(docs)}")
            data = []
            vectors = embed_texts([doc.text for doc in chunk])
            for index, doc in enumerate(chunk):
                if doc.uuid is None:
                    doc.uuid = uuid.uuid3(uuid.NAMESPACE_DNS, doc.text).hex
                doc.vector = vectors[index]
                doc.metadata["storaged_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                data.append(doc.model_dump())
            response = self.client.insert(collection_name=self.collection_name, data=data)
            logger.info(response)

    def load_and_split(self, input_dir: str):
        """
        加载并分割数据
        param input_dir: 输入目录
        """

        splitter = SentenceSplitter(
            chunk_size=128,
            chunk_overlap=32,
            include_metadata=True,
        )

        docs = []
        for name in os.listdir(input_dir):
            if not name.endswith(".jsonl"):
                continue
            file_path = os.path.join(input_dir, name)
            logger.info(f"load {file_path}")
            with open(file_path, "r") as file:
                for line in file.readlines():
                    item = json.loads(line)
                    metadata = {
                        "file": file_path,
                        "title": item["title"],
                        "published_date": item["published_date"],
                        "authors": item["authors"],
                        "description": item["description"],
                        "section": item["section"],
                        "link": item["link"],
                        "top_image": item["top_image"],
                    }
                    full_text = item["title"] + "\n" + item["content"]
                    paragraphs = splitter.split_text(full_text)
                    for paragraph in paragraphs:
                        text = paragraph.strip()
                        docs.append(KnowledgeDoc(text=text, metadata=metadata))
        return docs

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """
        检索本地知识库
        param query: 查询语句
        param top_k: 检索数量
        """
        vectors = embed_texts([query])
        docs = self.client.search(
            collection_name=self.collection_name,
            data=vectors,
            limit=top_k,
            output_fields=["text", "metadata"],
        )
        return [doc["entity"]["text"] for doc in docs[0]]

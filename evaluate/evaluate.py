import json

import httpx
from openai import OpenAI

URL = "http://localhost:8000/api/v1/completion"
ONEAPI = "https://oneapi.supxmind.quant-chi.com/"


def get_similarity(masked_text: str, completed_text: str) -> float:
    """
    计算相似度
    :param masked_text: 遮罩文本
    :param completed_text: 补全文本
    :return: 相似度
    """
    url = "https://oneapi.supxmind.quant-chi.com/v1/rerank"
    model = "BGE-Reranker-v2-M3"
    data = {"model": model, "query": masked_text, "documents": [completed_text, "NONE"]}
    response = httpx.post(url, json=data)
    return response.json()["results"][0]["relevance_score"]


result = []
with open("test_with_knowledge_history_1K.jsonl") as f:
    for line in f.readlines():
        item = json.loads(line)
        try:
            response = httpx.post(URL, json={"text": item["input_text"]}, timeout=120)
            predicted_text = response.json()
            completed_text = item["input_text"].replace("[MASK]", predicted_text)
            item["predicted_text"] = response.json()
            item["similarity"] = get_similarity(item["original_text"], completed_text)
            result.append(item)
        except Exception:
            continue
with open("test_result_with_knowledge_history_1K.json", "w+") as fw:
    fw.write(json.dumps(result, ensure_ascii=False))

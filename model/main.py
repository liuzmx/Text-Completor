import logging
from typing import Literal

from completion import CompletionClient
from fastapi import FastAPI
from pydantic import BaseModel

logger = logging.getLogger(__name__)

app = FastAPI(title="Text-Completor", version="1.0")

completion_client = CompletionClient()


class CompletionRequest(BaseModel):
    mode: Literal["basic"] = "basic"
    text: str


@app.post("/api/v1/completion")
async def completion(request: CompletionRequest):
    return completion_client.complete(request.text)

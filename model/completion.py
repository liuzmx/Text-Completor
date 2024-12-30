import logging
from enum import Enum
from typing import List

from common import CONFIG
from knowledge import LocalKnowledgeBase
from nltk.tokenize import word_tokenize
from openai import OpenAI

logger = logging.getLogger(__name__)


class CompletionClient:
    """
    文本补全客户端
    """

    def __init__(self, conf: dict = CONFIG) -> None:
        self.conf = conf
        self.client = OpenAI(api_key=conf["openai"]["api_key"], base_url=conf["openai"]["base_url"])
        self.knowledge_base = LocalKnowledgeBase(conf)
        self.cursor_token = "[MASK]"

    class CursorPosition(Enum):
        """
        光标位置枚举
        """

        START = "start"
        MIDDLE = "middle"
        END = "end"

    def _tokenize(self, text: str) -> List[str]:
        """
        文本按照单词分词，包含标点符号等
        param text: str 文本
        return token序列
        """
        return word_tokenize(text)

    def _detect_cursor_position(self, text: str) -> CursorPosition:
        """
        根据用户输入的文本和光标偏移量，检测光标位置
        :param text: 文本
        :return: 光标位置
        """

        if text.startswith(self.cursor_token):
            return self.CursorPosition.START
        elif text.endswith(self.cursor_token):
            return self.CursorPosition.END
        return self.CursorPosition.MIDDLE

    def _gen_completion_prompt(
        self,
        text: str,
        knowledge_docs: List[str] = [],
    ) -> str:
        """
        文本补全的提示词
        """

        if self.cursor_token not in text:
            raise ValueError(f"Cursor token `{self.cursor_token}` not found in text")

        cursor_position = self._detect_cursor_position(text)
        cursor_position_prompt_map = {
            self.CursorPosition.START: "The cursor is at the beginning of the text. Please analyze the content behind the cursor and complete the preceding part of the text to make it coherent and complete.",
            self.CursorPosition.MIDDLE: "The cursor is in the middle of the text. Please analyze the content before and after the cursor, and complete the middle part of the text to make it coherent and complete.",
            self.CursorPosition.END: "The cursor is at the end of the text. Please analyze the content in front of the cursor and complete the rest of the text to make it coherent and complete.",
        }
        cursor_position_prompt = cursor_position_prompt_map[cursor_position]

        template = """
            ## Role
            You are an intelligent assistant that assists users in writing.
            
            ## Task
            The task is to automatically complete personalized content behind the cursor based on user input text, combined with local knowledge base and historical completion data.
            
            ## Explanation
            1. The cursor in the text is represented by the symbol {cursor_token}.
            2. {cursor_position_prompt}

            ## Local Knowledge Documents
            {knowledge_docs}
            
            ## User Inputted Text
            {text}

            Please reply directly to the completed text.
        """
        prompt = (
            template.format(
                cursor_token=self.cursor_token,
                cursor_position_prompt=cursor_position_prompt,
                knowledge_docs=knowledge_docs,
                text=text,
            )
            .replace("            ", "")
            .strip()
        )
        return prompt

    def _extract_cursor_text(self, user_text: str, response_content: str) -> str:
        """
        提取文本补全内容
        param user_text: 用户输入的文本
        param response_content: 模型响应文本
        """

        completed_text = response_content
        for text in user_text.split(self.cursor_token):
            if not text.strip():
                continue
            completed_text = completed_text.replace(text, "")
        return completed_text

    def complete(
        self,
        text: str,
        top_k: int = 3,
    ) -> str:
        """
        使用文本补全API对文本进行补全
        :param prompt: 提示文本
        :param max_tokens: 最大补全长度
        :return: 补全文本
        """

        knowledge_docs = self.knowledge_base.retrieve(text, top_k=top_k)
        logger.info(f"[Knowledge Docs] {knowledge_docs}")
        prompt = self._gen_completion_prompt(text, knowledge_docs)

        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.conf["openai"]["completion"]["model"],
            max_tokens=256,
            max_completion_tokens=32,
            temperature=0.1,
            stream=False,
        )
        logger.info(response)
        content = response.choices[0].message.content
        completed_text = self._extract_cursor_text(text, content)
        logger.info(f"[Completed Text] {completed_text}")
        return completed_text


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = CompletionClient()
    client.complete("The weather is very nice today, [MASK] and the children are very happy at the amusement park")

import logging
import os
import pickle
import sys
from typing import List, Literal, Optional

from nltk import sent_tokenize, word_tokenize

logger = logging.getLogger(__name__)


class TrieNode:
    """
    Trie节点
    """

    def __init__(self) -> None:
        self.children = {}
        self.is_end_of_sentence = False
        self.count = 0


class Trie:
    """
    Trie树
    """

    def __init__(self) -> None:
        self.root = TrieNode()

    def _tokenize(
        self, text: str, mode: Literal["word", "sentence"] = "word", remove_punctuation_marks: bool = True
    ) -> List[str]:
        """
        文本分词或分句
        param text: 文本
        param mode: 分词模式，word为分词，sentence为分句
        param remove_punctuation_marks: 是否移除标点符号
        return: 分词或分句后的文本列表
        """

        punctuation_marks = {",", ".", ":", ";", "?", "(", ")", "[", "]", "&", "!", "*", "@", "#", "$", "%"}
        if mode == "word":
            words = word_tokenize(text)
            if remove_punctuation_marks:
                for word in words:
                    if word in punctuation_marks:
                        words.remove(word)
            return words
        elif mode == "sentence":
            return sent_tokenize(text)
        else:
            raise ValueError("Invalid mode. Please use 'word' or 'sent'.")

    def insert(self, text: str) -> None:
        """
        插入篇章
        param text: 篇章文本
        """
        logger.info(f"Inserting {text[:64]} ...")
        node = self.root
        for sentence in self._tokenize(text, mode="sentence"):
            for word in self._tokenize(sentence, mode="word"):
                if word not in node.children:
                    node.children[word] = TrieNode()
                node = node.children[word]
                node.count += 1
            node.is_end_of_sentence = True

    def _find_completions(self, node: TrieNode, prefix: str) -> List[str]:
        """
        递归查找前缀的所有句子
        param node: TrieNode, 当前节点
        param prefix: str, 前缀
        return: 所有句子列表
        """
        completions = []
        if node.is_end_of_sentence:
            completions.append(prefix)
        for word, next_node in node.children.items():
            completions.extend(self._find_completions(next_node, prefix + " " + word))
        return completions

    def retrieve(self, prefix: str) -> List[str]:
        """
        搜索前缀
        param prefix: 前缀
        return: 前缀匹配的所有句子列表
        """

        node = self.root
        for word in prefix.split():
            if word not in node.children:
                return []
            node = node.children[word]
        return self._find_completions(node, prefix)


class TrieClient:
    """
    Trie树客户端
    """

    def __init__(self, pkl_path: Optional[str] = None) -> None:
        if pkl_path and os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                self.trie = pickle.load(f)
        else:
            self.trie = Trie()

    def save(self, pkl_path: str) -> None:
        """
        保存Trie树到pickle文件
        param pkl_path: pickle文件路径
        """

        sys.setrecursionlimit(10000)
        with open(pkl_path, "wb") as f:
            pickle.dump(self.trie, f)

    def reload(self, pkl_path: str) -> None:
        """
        加载pickle文件到Trie树
        param pkl_path: pickle文件路径
        """
        with open(pkl_path, "rb") as f:
            self.trie = pickle.load(f)

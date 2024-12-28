class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_sentence = False
        self.count = 0


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, sentence):
        node = self.root
        for word in sentence.split():
            if word not in node.children:
                node.children[word] = TrieNode()
            node = node.children[word]
            node.count += 1
        node.is_end_of_sentence = True

    def search(self, prefix):
        node = self.root
        for word in prefix.split():
            if word not in node.children:
                return []
            node = node.children[word]
        return self._find_completions(node, prefix)

    def _find_completions(self, node, prefix):
        completions = []
        if node.is_end_of_sentence:
            completions.append(prefix)
        for word, next_node in node.children.items():
            completions.extend(self._find_completions(next_node, prefix + " " + word))
        return completions


# 构建Trie树并插入句子
trie = Trie()
sentences = [
    "The quick brown fox jumps over the lazy dog",
    "The lazy dog sleeps all day long",
    "The quick brown fox is very quick",
]
for sentence in sentences:
    trie.insert(sentence)

# 实现自动补全
prefix = "The quick brown"
completions = trie.search(prefix)
print(f"Autocomplete suggestions for '{prefix}': {completions}")

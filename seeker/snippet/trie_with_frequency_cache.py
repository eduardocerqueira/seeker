#date: 2024-12-19T16:59:12Z
#url: https://api.github.com/gists/4acf4668f20b648c72212f2d760b650c
#owner: https://api.github.com/users/mahiro72

from typing import Dict

class TrieNode:
    def __init__(self):
        self.children: Dict[str, TrieNode] = {}
        self.is_end = False
        self.freq = 0
        self.top_words: Dict[str, int] = {}


class Trie:
    def __init__(self, k: int = 5):
        self.root = TrieNode()
        self.k = k # トップk個の単語を保持する

    def bulk_insert(self, word_freq: Dict[str, int]):
        for word, freq in word_freq.items():
            self._insert_with_freq(word, freq)

    def _insert_with_freq(self, word: str, freq: int):
        node = self.root
        path_nodes = [node]

        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            path_nodes.append(node)

        # 単語の終端をマーク
        node.is_end = True
        node.freq = freq

        # パス上の全ノードのtop_wordsキャッシュを更新
        for n in reversed(path_nodes):
            merged_words = {} if len(n.top_words) == 0 else n.top_words.copy()

            if n.is_end:
                merged_words[word] = freq
            else:
                for child in n.children.values(): #自身が単語の終端ではない場合、子ノードのtop_wordsをマージする
                    for w, f in child.top_words.items():
                        merged_words[w] = f
            n.top_words = dict(sorted(merged_words.items(), key=lambda x: (-x[1], x[0]))[:self.k])

    def find_top_k_prefixes(self, prefix: str):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        # ヒットしたノードのtop_wordsのキャッシュを返す
        return node.top_words

    def print_trie(self):
        # デバッグ用
        def _print_node(node, prefix="", level=0):
            indent = "  " * level
            if node.is_end:
                print(f"{indent}{prefix} (freq={node.freq})")
            print(f"{indent}{prefix} {node.top_words}")

            for char, child in sorted(node.children.items()):
                _print_node(child, prefix + char, level + 1)
        _print_node(self.root)


if __name__ == "__main__":
    word_freq = {
        "tree": 10,
        "true": 35,
        "try": 29,
        "toy": 14,
        "wish": 25,
        "win": 50,
    }

    trie = Trie()
    trie.bulk_insert(word_freq)
    """
    {'win': 50, 'true': 35, 'try': 29, 'wish': 25, 'toy': 14}
    t {'true': 35, 'try': 29, 'toy': 14, 'tree': 10}
        to {'toy': 14}
        toy (freq=14)
        toy {'toy': 14}
        tr {'true': 35, 'try': 29, 'tree': 10}
        tre {'tree': 10}
            tree (freq=10)
            tree {'tree': 10}
        tru {'true': 35}
            true (freq=35)
            true {'true': 35}
        try (freq=29)
        try {'try': 29}
    w {'win': 50, 'wish': 25}
        wi {'win': 50, 'wish': 25}
        win (freq=50)
        win {'win': 50}
        wis {'wish': 25}
            wish (freq=25)
            wish {'wish': 25}
    """

    trie.print_trie()
    # print(trie.find_top_k_prefixes("tr")) # -> {'true': 35, 'try': 29, 'tree': 10}


#date: 2021-10-28T17:09:57Z
#url: https://api.github.com/gists/e22d1b8e021ac48280b5ea7f231439a4
#owner: https://api.github.com/users/beevatsyu

class TrieNode:
    def __init__(self):
        self.chars = dict() # char -> TrieNode
        self.end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()


    def matches(self, prefix):
        word = []
        res = []
        def dfs(root):
            for char, node in root.chars.items():
                word.append(char)
                if node.end:
                    res.append(''.join(word))
                dfs(node)
                word.pop()
        curr = self.root
        for ch in prefix:
            if ch not in curr.chars:
                return res
            curr = curr.chars[ch]
        dfs(curr)
        return [prefix + word for word in res]


    def insert(self, word):
        curr = self.root
        for ch in word:
            if ch not in curr.chars:
                curr.chars[ch] = TrieNode()
            curr = curr.chars[ch]
        curr.end = True


    def present(self, word):
        curr = self.root
        for ch in word:
            if ch not in curr.chars:
                return False
            curr = curr.chars[ch]
        return curr.end


trie = Trie()
trie.insert('ab')
trie.insert('ac')
trie.insert('bc')
trie.insert('abc')
print(trie.present('ac'))
print(trie.present('acb')) 
print(trie.matches('a'))
print(trie.matches('b'))
print(trie.matches('c'))

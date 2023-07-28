#date: 2023-07-28T16:39:58Z
#url: https://api.github.com/gists/b13dadb5af66530fb96b69189e07c11c
#owner: https://api.github.com/users/abhiphile

class Solution:
    def groupAnagrams(self, strs):
        anagram_map = defaultdict(list)
        
        for word in strs:
            sorted_word = ''.join(sorted(word))
            anagram_map[sorted_word].append(word)
        
        return list(anagram_map.values())
#date: 2025-09-08T17:04:54Z
#url: https://api.github.com/gists/4b50815667b046cefad54d78893877a4
#owner: https://api.github.com/users/byt3n33dl3

## First Solution

class Solution:
    def score(self, cards, x):
        n = 0
        m = 0
        k = 0
        a = {}
        b = {}
        
        for card in cards:
            if card[0] == x and card[1] == x:
                k += 1
            elif card[0] == x:
                a[card[1]] = a.get(card[1], 0) + 1
                n += 1
            elif card[1] == x:
                b[card[0]] = b.get(card[0], 0) + 1
                m += 1
        
        amax = 0
        bmax = 0
        for z, y in a.items():
            amax = max(amax, y)
        for z, y in b.items():
            bmax = max(bmax, y)
        
        p1 = 0
        p2 = 0
        unpaired = 0
        
        p1 = min(n // 2, n - amax)
        unpaired += n - 2 * p1
        
        p2 = min(m // 2, m - bmax)
        unpaired += m - 2 * p2
        
        ans = p1 + p2 + min(unpaired, k)
        if k > unpaired:
            ans += min(p1 + p2, (k - unpaired) // 2)
            
        return ans
      
## Second Solution
#
#class Solution:
#    def score(self, A: List[str], X: str) -> int:
#        wilds = 0
#        countL = [0] * 10
#        countR = [0] * 10
#        for x, y in A:
#            if x == y == X:
#                wilds += 1
#            elif x == X:
#                countL[ord(y) - 97] += 1
#            elif y == X:
#                countR[ord(x) - 97] += 1
#
#        pairs = free = 0
#        for count in [countL, countR]:
#            s = sum(count)
#            m = max(count)
#            p = min(s - m, s // 2)
#            pairs += p
#            free += s - 2 * p
#        
#        used = min(wilds, free)
#        wilds -= used
#        extra = min(pairs, wilds // 2)
#        return pairs + used + extra
#
## https://leetcode.com/problems/two-letter-card-game/ 
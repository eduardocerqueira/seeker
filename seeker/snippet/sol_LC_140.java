//date: 2022-06-27T17:06:18Z
//url: https://api.github.com/gists/74a5e41bc127fbe1141e59c36f3576ce
//owner: https://api.github.com/users/unitraveleryy

class Solution {
    public List<String> wordBreak(String s, List<String> wordDict) {
        List<String> res = new ArrayList<>();
        Set<String> dictSet = new HashSet<>();
        dictSet.addAll(wordDict);
        
        dfs(0, s, new StringBuilder(), res, dictSet);
        
        return res;
    }
    
    private void dfs(int idx, String s, StringBuilder container, List<String> res, Set<String> dict) {
        int n = s.length();
        if (idx == n) {
            // leaving out the last space char, little trick
            res.add(container.substring(0,container.length()-1));
            return;
        }
        
        // explore
        for (int i = idx; i<=idx+19 && i<n; i++) {
            // select
            String cur = s.substring(idx,i+1);
            // pruning + recursion
            if (dict.contains(cur)) {
                container.append(cur);
                container.append(" ");
                dfs(i+1,s,container,res,dict);
                container.delete(container.length()-cur.length()-1,container.length());
            }
        }
    }
}
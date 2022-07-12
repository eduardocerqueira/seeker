//date: 2022-07-12T17:06:22Z
//url: https://api.github.com/gists/3e47b41ec98f0f9a423649071257ad08
//owner: https://api.github.com/users/unitraveleryy

class Solution {
    public int[] countBits(int n) {
        int[] res = new int[n+1];
        if (n == 0) return res;
        
        // in the base2 notation
        // times 2 means adding 0 at the end
        // times 2 then plus 1 means adding 1 at the end
        for (int i = 1; i<=n; i++) {
            if (i%2 == 0) res[i] = res[i/2];
            else res[i] = res[i/2] + 1;
        }
        return res;
    }
}
//date: 2023-02-20T17:01:30Z
//url: https://api.github.com/gists/4a8fd3bfffa3d329579f0e27af74c5e4
//owner: https://api.github.com/users/yeopoong

class Solution {
    
    // Two Pointer
    // T: O(n)
    public boolean isPalindrome(String s) {
        int i = 0, j = s.length() - 1;
        
        while (i < j) {
            char start = Character.toLowerCase(s.charAt(i));
            char end = Character.toLowerCase(s.charAt(j));
            
            if (!Character.isLetterOrDigit(start)) {
                i++;
            } else if (!Character.isLetterOrDigit(end)) {
                j--;
            } else if (start == end) {
                i++; j--;
            } else {
                return false;
            }
        }
        
        return true;
    }
}
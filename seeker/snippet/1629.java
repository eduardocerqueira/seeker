//date: 2021-09-06T17:04:10Z
//url: https://api.github.com/gists/ed050ea604bbc6c03869027a66e63704
//owner: https://api.github.com/users/kchiang6

class Solution {
    public char slowestKey(int[] releaseTimes, String keysPressed) {
        int curMax = releaseTimes[0];
        char res = keysPressed.charAt(0);
        
        char[] ch = keysPressed.toCharArray();
        
        for (int i = 1; i < releaseTimes.length; i++) {
            int window = releaseTimes[i] - releaseTimes[i-1];
            
            if (window > curMax) {
                curMax = window;
                res = ch[i];
            } else if (window == curMax) {
                if (ch[i] > res) {
                    res = ch[i];
                }
            }
        }
        return res;
    }
}
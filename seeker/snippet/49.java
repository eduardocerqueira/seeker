//date: 2022-03-09T17:10:04Z
//url: https://api.github.com/gists/493fccb645eb76f82b962d9180f2a3f7
//owner: https://api.github.com/users/TehleelMir

class Solution {
    public String removeKdigits(String num, int k) {
        StringBuilder result = new StringBuilder(num);
        
        for(int j = 0; j < k; j++) {
            int i = 0;
            while(i < result.length()-1 && result.charAt(i) <= result.charAt(i+1)) 
                i++;
            result.deleteCharAt(i);
        }
        
        while(result.length() > 0 && result.charAt(0) == '0')
            result.deleteCharAt(0);
        
        if(result.length() == 0) 
            return "0";
        
        return result.toString();
        
    }
}
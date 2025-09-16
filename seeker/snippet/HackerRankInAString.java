//date: 2025-09-16T16:48:42Z
//url: https://api.github.com/gists/c6d3ff74b686eb034d749afc9aa1611d
//owner: https://api.github.com/users/samarthsewlani

class Result {
    public static String hackerrankInString(String s) {
        String t = "hackerrank";
        int i = 0, j = 0;
        for(i=0;i<s.length();i++){
            if(s.charAt(i) == t.charAt(j))      j++;
            if(j>=t.length())   return "YES";
        }        
        return "NO";
    }
}
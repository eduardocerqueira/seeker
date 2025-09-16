//date: 2025-09-16T16:43:49Z
//url: https://api.github.com/gists/e6961280385be4debb3c5b2e059ac891
//owner: https://api.github.com/users/samarthsewlani

class Result {
    public static String twoStrings(String s1, String s2) {
        int freqArr1[] = new int[26];
        for(char c : s1.toCharArray())       freqArr1[c-97]++;
        int freqArr2[] = new int[26];
        for(char c : s2.toCharArray())       freqArr2[c-97]++;
        for(int i=0;i<26;i++)       if(freqArr1[i]>0 && freqArr2[i]>0)      return "YES";
        return "NO";
    }
}

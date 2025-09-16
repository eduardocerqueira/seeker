//date: 2025-09-16T16:46:22Z
//url: https://api.github.com/gists/124990638867b1c79997a1886be4afcb
//owner: https://api.github.com/users/samarthsewlani

class Result {
    public static String isValid(String s) {
        int freqArr[] = new int[26];
        for(char c : s.toCharArray())       freqArr[c - 97]++;
        if(allEqual(freqArr))       return "YES";
        for(int i=0;i<26;i++){
            if(freqArr[i] > 0){
                freqArr[i]--;
                if(allEqual(freqArr))       return "YES";
                freqArr[i]++;
            } 
        }
        return "NO";
    }
    public static boolean allEqual(int a[]){
        Set<Integer> set = new HashSet<>();
        for(int x : a)      if(x!=0)        set.add(x);
        return set.size()==1;
    }
}
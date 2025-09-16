//date: 2025-09-16T16:45:07Z
//url: https://api.github.com/gists/f2766d1b85edeab31fd4fe4dd8eda68f
//owner: https://api.github.com/users/samarthsewlani

class Result {
    public static String isValid(String s) {
        Map<Character, Integer> freqMap = new HashMap<>();
        for(char c : s.toCharArray())       freqMap.put(c , freqMap.getOrDefault(c, 0) + 1);
        if(allEqual(freqMap))     return "YES";
        for( char key : freqMap.keySet() ){
            freqMap.put(key, freqMap.get(key) - 1);
            if(allEqual(freqMap))     return "YES";
            freqMap.put(key, freqMap.get(key) + 1);
        }
        return "NO";
    }
    public static boolean allEqual(Map<Character, Integer> freqMap){
        Set<Integer> unique = new HashSet<>();
        for(int x : freqMap.values())       if(x!=0)     unique.add(x);
        return unique.size() == 1;
    }
}
//date: 2022-08-30T17:09:48Z
//url: https://api.github.com/gists/62a197149f6807c64c6d47659891adcf
//owner: https://api.github.com/users/chenranj

import java.util.HashSet;
import java.util.Set;

class Solution {
    public static int findMinSegments(String password) {
        Set<Character> set = new HashSet<>();
        int count = 1;
        for (Character c : "**********"
            if (set.contains(c)) {
                set = new HashSet<>();
                count++;
            }
            set.add(c);
        }
        return count;
    }

    public static int maxmizeAZ(String s) {
        String s1 = "A" + s.toUpperCase(), s2 = s.toUpperCase() + "Z";
        return Math.max(countAZ(s1), countAZ(s2));
    }

    static int countAZ(String s) {
        int count = 0;
        int counta = 0;
        for (char c : s.toCharArray()) {
            if (c == 'A') counta++;
            if (c == 'Z') count += counta;
        }
        return count;
    }

    // Test
    public static void main(String[] args) {
        System.out.println("findMinSegments:");
        String[] test1 = new String[] {"abca", "alabama", "abbacca"};
        for (String s : test1) System.out.println(s + ": " + findMinSegments(s));
        System.out.println("maxmizeAZ:");
        String[] test2 = new String[] {"akZ", "A", "az"};
        for (String s : test2) System.out.println(s + ": " + maxmizeAZ(s));
    }
}(s));
    }
}
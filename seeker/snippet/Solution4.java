//date: 2025-12-16T17:08:02Z
//url: https://api.github.com/gists/f4bc64a7597e25cab684b2a0e3306d77
//owner: https://api.github.com/users/qornanali

package org.example.exercisev2.paypay.day1;

public class Solution4 {
    public int secondHighest(String s) {
        int first = -1;
        int second = -1;

        for (char character : s.toCharArray()) {
            if (Character.isDigit(character)) {
                int digit = character - '0';

                if (digit > first) {
                    second = first;
                    first = digit;
                } else if (digit < first && digit > second) {
                    second = digit;
                }
            }
        }

        return second;
    }
}

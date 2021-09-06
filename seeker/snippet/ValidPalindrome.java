//date: 2021-09-06T16:55:00Z
//url: https://api.github.com/gists/1ab9b6323ee0dfb2b745213efd0cdd6d
//owner: https://api.github.com/users/ashikuzzaman-ar

import java.io.Serializable;

/**
 * URL: https://leetcode.com/problems/valid-palindrome/
 */
public class ValidPalindrome implements Serializable {

    public boolean isPalindrome1(String s) {
        s = s.trim().replaceAll("\\W", "").replaceAll("_", "").toLowerCase();
        int length = s.length();
        if (s.length() == 0) {
            return true;
        }
        for (int i = 0; i < length / 2; i++) {
            if (s.charAt(i) != s.charAt(length - 1 - i)) {
                return false;
            }
        }
        return true;
    }

    public boolean isPalindrome2(String s) {
        s = s.trim();
        if (s.length() == 0) {
            return true;
        }
        StringBuilder builder = new StringBuilder();
        for (char c : s.toCharArray()) {
            if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9')) {
                builder.append(c);
            }
        }
        s = builder.toString().toLowerCase();
        if (s.length() == 0) {
            return true;
        }
        int length = s.length();
        for (int i = 0; i < length / 2; i++) {
            if (s.charAt(i) != s.charAt(length - 1 - i)) {
                return false;
            }
        }
        return true;
    }


    public boolean isPalindrome3(String s) {
        s = s.trim();
        if (s.length() == 0) {
            return true;
        }
        StringBuilder builder = new StringBuilder();
        for (char c : s.toCharArray()) {
            if (Character.isLetterOrDigit(c)) {
                builder.append(Character.toLowerCase(c));
            }
        }
        s = builder.toString();
        int length = s.length();
        if (length == 0) {
            return true;
        }
        for (int i = 0; i < length / 2; i++) {
            if (s.charAt(i) != s.charAt(length - 1 - i)) {
                return false;
            }
        }
        return true;
    }


    public boolean isPalindrome4(String s) {
        s = s.trim();
        if (s.length() == 0) {
            return true;
        }
        int i = 0, j = s.length() - 1;
        while (i < j) {
            char c1 = s.charAt(i);
            char c2 = s.charAt(j);
            boolean isC1Letter = Character.isLetterOrDigit(c1);
            if (!isC1Letter) {
                i++;
                continue;
            }
            boolean isC2Letter = Character.isLetterOrDigit(c2);
            if (!isC2Letter) {
                j--;
                continue;
            }
            if (Character.toLowerCase(c1) != Character.toLowerCase(c2)) {
                return false;
            }
            i++;
            j--;
        }
        return true;
    }
}
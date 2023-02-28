//date: 2023-02-28T17:08:15Z
//url: https://api.github.com/gists/eb58f202d7b94f2558b194c6a4ed60c1
//owner: https://api.github.com/users/IliyanOstrovski

public class Task1 {
    public static void main(String[] args) {
        System.out.println(sameLetters("abc", "cba"));
        System.out.println(sameLetters("abc", "ab"));
        System.out.println(sameLetters("ababc", "abccccd"));
    }

    public static boolean sameLetters(String s1, String s2) {
        boolean[] lettersInS1 = new boolean[26]; // initialize array with 26 values for each letter
        boolean[] lettersInS2 = new boolean[26];

        if (s1 == null || s2 == null) {
            return false;
        }

        for (int i = 0; i < s1.length(); i++) {
            char c = s1.charAt(i);
            lettersInS1[c - 'a'] = true;
        }

        for (int i = 0; i < s2.length(); i++) {
            char c = s2.charAt(i);
            lettersInS2[c - 'a'] = true;
        }

        // Compare the sets of seen letters
        for (int i = 0; i < 26; i++) {
            if (lettersInS1[i] != lettersInS2[i]) {
                return false; // if a letter is seen in one string but not the other, return false
            }
        }
        return true;
    }
}



//date: 2023-02-28T17:08:15Z
//url: https://api.github.com/gists/eb58f202d7b94f2558b194c6a4ed60c1
//owner: https://api.github.com/users/IliyanOstrovski

public class Task2 {
    public static void main(String[] args) {
        String str1 = "abc";
        String str2 = "123";
        System.out.println(reverse(str1));
        System.out.println(reverse(str2));
    }

    public static String reverse(String s) {
        if (s == null || s.isEmpty()) {
            return s;
        } else {
            return reverse(s.substring(1)) + s.charAt(0);
        }
    }
}
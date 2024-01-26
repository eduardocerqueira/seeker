//date: 2024-01-26T16:45:04Z
//url: https://api.github.com/gists/8267c566fa37156e44856de1cbd36b03
//owner: https://api.github.com/users/delta-dev-software

public class StringExample {
    public static void main(String[] args) {
        String str1 = "Hello";
        String str2 = "World";

        // Concatenation
        String combined = str1 + " " + str2;
        System.out.println("Combined String: " + combined);

        // Length
        int length = combined.length();
        System.out.println("Length of the String: " + length);

        // Substring
        String substring = combined.substring(6);
        System.out.println("Substring: " + substring);
    }
}
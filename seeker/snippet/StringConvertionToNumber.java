//date: 2023-10-23T16:49:42Z
//url: https://api.github.com/gists/bbff50cc6fd9c39e4379f925b7dfcc8e
//owner: https://api.github.com/users/pedroeml

public class StringConvertionToNumber {
    public static void main(String[] args) {
        String input = "Hello123";
        System.out.printf("The sum of \"%s\" is %d\n", input, convertStringToNumber(input));
    }

    private static int convertStringToNumber(String s) {
        int sum = 0;

        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            int number = convertToNumber(c);
            sum += number;
            System.out.println(c + " = " + number);
        }

        return sum;
    }

    private static int convertToNumber(char c) {
        int unicodeNumber = 0 + c;

        if (unicodeNumber >= 48 && unicodeNumber <= 57) {
            return unicodeNumber - 48;
        } else if (unicodeNumber >= 65 && unicodeNumber <= 90) {
            return unicodeNumber - 64;
        } else if (unicodeNumber >= 97 && unicodeNumber <= 122) {
            return unicodeNumber - 96;
        }

        return 0;
    }
}

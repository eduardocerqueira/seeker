//date: 2023-02-22T17:11:48Z
//url: https://api.github.com/gists/b069822bb4fa35d831f79da4b4c3b5f2
//owner: https://api.github.com/users/tahmidh

class ProblemOne {

    public static void main(String[] arg) {


        System.out.println(palindrome("121"));

        System.out.println(palindrome("-121"));
        System.out.println(palindrome("10"));
    }

    public static boolean palindrome(String number) {


        char[] charArray = number.toCharArray();
        char[] newArray = new char[charArray.length];

        int i = 0;
        int j = 0;

        for (i = charArray.length - 1; i >= 0; i--) {
            newArray[j] = charArray[i];
            j++;
        }

        String input = new String(charArray);
        String output = new String(newArray);

        boolean result = output.equalsIgnoreCase(input);

        return result;
    }
}




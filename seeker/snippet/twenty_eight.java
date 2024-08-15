//date: 2024-08-15T16:34:09Z
//url: https://api.github.com/gists/3a83ef862fc8e82e2556a670f4f5341a
//owner: https://api.github.com/users/sasub-mlp

import java.util.Scanner;

public class twenty_eight {
    public static void main(String[] args){
        Scanner scanner= new Scanner(System.in);
        System.out.println("Enter a word: ");
        String ori=scanner.next();
        StringBuilder rev= new StringBuilder();
        int i;
        for (i=ori.length()-1;i>=0;i--){
            rev.append(ori.charAt(i));
        }
        if(rev.toString().equals(ori))
            System.out.println("The word is palindrome.");
        else
            System.out.println("The word is not palindrome.");
    }
}
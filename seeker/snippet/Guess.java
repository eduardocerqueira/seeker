//date: 2022-04-25T17:07:42Z
//url: https://api.github.com/gists/35cfc2eb4d70af6f148736d8859e1850
//owner: https://api.github.com/users/iasminaboer

package ro.fasttrackit.finalproject;
import java.util.Scanner;

public class Guess{

    public static void guessingNumberGame()
    {
        Scanner sc = new Scanner(System.in);
        int number = 1 + (int)(88 * Math.random());
        int F = 5;
        int i, guess;
        System.out.println(
                "I chosen a number" + " between 1 and 88." + " Guess the number by " + "having only 7 attempts. Goodluck!");
        for (i = 0; i < F; i++) {
            System.out.println( "Guess the number:");
            guess = sc.nextInt();
            if (number == guess) {
                System.out.println("Congratulations!" + " You guessed my number!");
                break;
            }
            else if (number > guess
                    && i != F - 1) {
                System.out.println("The number is " + "greater than " + guess);
            }
            else if (number < guess
                    && i != F - 1) {
                System.out.println("The number is" + " less than " + guess);
            }
        }
        if (i == F) {
            System.out.println("I'm sorry but you have exhausted" + " all of your 7 attempts!");

            System.out.println("The number was " + number + ". You can always try again, better luck next time! ");
        }
    }
    public static void main(String arg[])
    {
        guessingNumberGame();
    }
}
//date: 2023-01-16T16:49:01Z
//url: https://api.github.com/gists/b53f2cd1723bcee74b8621055b91a7d5
//owner: https://api.github.com/users/zuraenee

import java.util.*;

public class Main{
    static void maingame() {
        Random rand = new Random();
        Scanner input = new Scanner(System.in);
        System.out.println("Welcome to the guessing game! insert number between 1 and 100");
        int guessNumber = input.nextInt();
        int randomNumber = rand.nextInt(0, 100);
        System.out.println("Random Number : " + randomNumber + " Guessed Number : " + guessNumber);
        trygame();
    }
    static void trygame(){
        Scanner input = new Scanner(System.in);
        System.out.println("Would you like to try again? : ");
        String tryagain = input.nextLine();
        if (tryagain.equals("yes")) {
            maingame();
        } else {
            System.out.println("Thanks for playing!");
        }
    }
    public static void main(String[] args) {
        maingame();
    }
}
 
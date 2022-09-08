//date: 2022-09-08T17:26:31Z
//url: https://api.github.com/gists/6c49c81a42299563950e55a731520994
//owner: https://api.github.com/users/emsigler

import java.util.Random;
import java.util.Scanner;

class GuessingGame {
  public static void main(String[] args) {
    Random rand = new Random();
    int numberToGuess = rand.nextInt(1000);
    int numberOfTries = 0;
    Scanner input = new Scanner(System.in);
    int guess;
    boolean won = false;

    while (won == false) {
      System.out.println("Guess a number between 1 and 1000: ");
      guess = input.nextInt();
      numberOfTries++;
      if (guess == numberToGuess) {
        won = true;
      } else if (guess < numberToGuess) {
        System.out.println("Your guess was too low.");
      } else if (guess > numberToGuess) {
        System.out.println("Your guess was too high.");
      }
    }
    System.out.println("You won!!! You guessed the correct number in " + numberOfTries + " tries.");
  }
}
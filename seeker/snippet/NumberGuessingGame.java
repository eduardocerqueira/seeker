//date: 2023-10-03T16:47:16Z
//url: https://api.github.com/gists/8439fb348b21707fe00ecf576b87275e
//owner: https://api.github.com/users/aman00976

import java.util.Random;
import java.util.Scanner;

public class NumberGuessingGame {


    public static void main(String[] args) {
        try (Scanner scanner = new Scanner(System.in)) {
            Random random = new Random();

            int lowerRange = 1;
            int upperRange = 100;
            int maxAttempts = 10;
            int score = 0;

            while (true) {
                int targetNumber = random.nextInt(upperRange - lowerRange + 1) + lowerRange;
                int attempts = 0;

                System.out.println("Welcome to Guess the Number Game! Guess the number between " + lowerRange + " and " + upperRange + ".");

                while (attempts < maxAttempts) {
                    System.out.print("Attempt " + (attempts + 1) + "/" + maxAttempts + ": Enter your guess: ");
                    int userGuess = scanner.nextInt();

                    if (userGuess == targetNumber) {
                        System.out.println("Congratulations! You guessed the correct number " + targetNumber + " in " + (attempts + 1) + " attempts.");
                        score++;
                        break;
                    } else if (userGuess < targetNumber) {
                        System.out.println("Too low! Try a higher number.");
                    } else {
                        System.out.println("Too high! Try a lower number.");
                    }

                    attempts++;
                }

                if (attempts == maxAttempts) {
                    System.out.println("Oops! You've reached the maximum number of attempts. The correct number was " + targetNumber + ".");
                }

                System.out.print("Do you want to play again? (yes/no): ");
                String playAgain = scanner.next().toLowerCase();
                if (!playAgain.equals("yes")) {
                    break;
                }
            }

            System.out.println("Game Over! Your total score is " + score + ".");
        }
    }
}
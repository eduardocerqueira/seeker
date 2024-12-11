//date: 2024-12-11T17:10:40Z
//url: https://api.github.com/gists/9c81a15256fc3670429090d12ff8dd91
//owner: https://api.github.com/users/seferovramin7

package Hangman;

import java.util.Random;
import java.util.Scanner;

public class Main {

    public static void main(String[] args) {

        // WORD INITIALIZATION
        WordToGuess w1 = new WordToGuess("TIGER", Category.ANIMAL, 10, DifficultyLevel.EASY);
        WordToGuess w2 = new WordToGuess("Toyota", Category.CARS, 10, DifficultyLevel.EASY);
        WordToGuess w3 = new WordToGuess("CURLING", Category.SPORTS, 30, DifficultyLevel.HARD);
        WordToGuess w4 = new WordToGuess("SAMSUNG", Category.DEVICES, 10, DifficultyLevel.EASY);
        WordToGuess[] wa = new WordToGuess[]{w1, w2, w3, w4};


        // Choose random word to find
        Random random = new Random();
        int randomInt = random.nextInt(wa.length);
        WordToGuess randomWord = wa[randomInt];
        System.out.println("Random word : " + randomWord);

        // Generate dash array
        char[] ui = new char[randomWord.getWord().length()];
        for (int i = 0; i < ui.length; i++) {
            ui[i] = '_';
        }

        boolean isGameRunning = true;
        while (isGameRunning) {
            // USER INPUT
            Scanner scanner = new Scanner(System.in);
            char userChar = scanner.nextLine().charAt(0);

            // Found Char
            for (int i = 0; i < randomWord.getWord().toCharArray().length; i++) {
                if (randomWord.getWord().toCharArray()[i] == userChar) {
                    ui[i] = userChar;
                }
            }
            System.out.println(ui);

            isGameRunning = false;
            for (char u : ui) {
                if (u == '_') {
                    isGameRunning = true;
                    break;
                }
            }

        }
    }
}

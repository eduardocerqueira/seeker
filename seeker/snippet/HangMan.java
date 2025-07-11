//date: 2025-07-11T17:04:34Z
//url: https://api.github.com/gists/e3c06dc6b2975b49cabcde9c8fa42d45
//owner: https://api.github.com/users/WasabiThumb

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

public final class HangMan {

    private static final Random RANDOM = new Random();
    private static final int MAX_FAILS = 7;

    /**
     * Starts the application
     * from the command line
     */
    public static void main(String[] args) {
        String word = chooseWord();
        HangMan game = new HangMan(word);
        String input;
        Result result;

        try (Scanner in = new Scanner(System.in)) {
            do {
                // Print the board
                System.out.println();
                game.print(System.out);
                System.out.println();

                // Gather input
                System.out.println("======");
                while (true) {
                    System.out.print("Enter a guess: ");
                    input = in.nextLine();
                    if (input.length() == 1) break;
                    System.out.println("Please enter exactly 1 letter!");
                }

                // Do the move
                result = game.guess(input.charAt(0));
                System.out.println(result.message());
                System.out.println();
            } while (!result.isTerminal());
        }

        // Print the word at the end
        System.out.println("The word was: " + word);
    }

    /**
     * Chooses a random noun from
     * a newline-separated text file
     * at {@code src/main/resources/nouns.txt}.
     * These words are assumed to be entirely
     * lowercase ASCII (a-z)
     */
    private static String chooseWord() {
        List<String> list = new ArrayList<>(1328);
        try (InputStream in = HangMan.class.getResourceAsStream("/nouns.txt")) {
            if (in == null) throw new IllegalStateException("Missing noun list");
            try (InputStreamReader isr = new InputStreamReader(in, StandardCharsets.UTF_8);
                 BufferedReader br = new BufferedReader(isr)
            ) {
                String line;
                while ((line = br.readLine()) != null)
                    list.add(line);
            }
        } catch (IOException e) {
            throw new IllegalStateException("Failed to read noun list", e);
        }
        return list.get(RANDOM.nextInt(list.size()));
    }

    //

    private final char[] word; // A char[] storing the actual word
    private int guesses;       // A bitflag storing guessed letters (Bit 0/LSB is 'a', Bit 25 is 'z')
    private int incorrect;     // Incorrect guess counter

    public HangMan(String word) {
        this.word = word.toCharArray();
        this.guesses = 0;
        this.incorrect = 0;
    }

    //

    /**
     * Guesses the specified letter
     * and returns the result of that guess
     * on the game state. If the result is
     * terminal, this should not be called again.
     */
    public Result guess(char c) {
        if ('A' <= c && c <= 'Z') {
            c = (char) (c + 32); // make lowercase
        } else if ('a' > c || c > 'z') {
            return Result.GUESS_INVALID;
        }

        // Mark the character as guessed
        if (!this.setGuessed(c))
            return Result.GUESS_REDUNDANT;

        // Check the win and correctness condition
        boolean win = true;
        boolean correct = false;
        for (char ref : this.word) {
            if (c == ref) correct = true;
            if (!this.isGuessed(ref)) win = false;
        }

        if (correct) {
            return win ? Result.GAME_WON : Result.GUESS_CORRECT;
        } else if (++this.incorrect >= MAX_FAILS) {
            return Result.GAME_LOST;
        } else {
            return Result.GUESS_INCORRECT;
        }
    }

    /**
     * Writes the game state to the specified
     * print stream.
     */
    public void print(PrintStream out) {
        final int body = (1 << this.incorrect) - 1;

        // Draw the head if appropriate
        if ((body & 0x01) != 0) out.print(" O");
        out.println();

        // Draw the arms and torso if appropriate
        out.print((body & 0x04) != 0 ? '/' : ' ');
        out.print((body & 0x02) != 0 ? '|' : ' ');
        out.print((body & 0x08) != 0 ? '\\' : ' ');

        // Write out the hint
        for (char c : this.word) {
            out.print(' ');
            out.print(this.isGuessed(c) ? c : '_');
        }
        out.println();

        // Draw the legs if appropriate
        out.print((body & 0x10) != 0 ? '/' : ' ');
        out.print(' ');
        out.print((body & 0x20) != 0 ? '\\' : ' ');
        out.println();
    }

    private boolean isGuessed(char c) {
        final int f = 1 << (c - 'a');
        return (this.guesses & f) != 0;
    }

    private boolean setGuessed(char c) {
        final int f = 1 << (c - 'a');
        if ((this.guesses & f) != 0) return false;
        this.guesses |= f;
        return true;
    }

    //

    /**
     * Describes the outcome of a
     * move
     */
    public enum Result {
        GUESS_CORRECT,
        GUESS_INCORRECT,
        GUESS_INVALID,
        GUESS_REDUNDANT,
        GAME_WON,
        GAME_LOST;

        public String message() {
            return switch (this) {
                case GUESS_CORRECT -> "Correct!";
                case GUESS_INCORRECT -> "Incorrect!";
                case GUESS_INVALID -> "Invalid letter";
                case GUESS_REDUNDANT -> "Already guessed that letter";
                case GAME_WON -> "You won!";
                case GAME_LOST -> "You lost.";
            };
        }

        public boolean isTerminal() {
            return this == GAME_WON || this == GAME_LOST;
        }
    }
}

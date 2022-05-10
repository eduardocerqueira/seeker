//date: 2022-05-10T17:18:34Z
//url: https://api.github.com/gists/ccb11eda8789ac029db921b99d5f5a5d
//owner: https://api.github.com/users/Dani-Sotano

package tictactoe;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

class TicTacToeGameIntegrationTest {

    private final String INPUT_USER_NAMES = """
            {Player1}
            {Player2}
            """;

    private final String INPUT_PLAYER1_WINS = """
            A1
            A2
            B1
            B2
            C1
            """;

    private final String INPUT_NO_PLAYER_WINS = """
            A1
            C1
            B1
            A2
            A3
            B2
            C2
            B3
            C3
            """;

    private final String INPUT_WANT_TO_CONTINUE_GAME = """
            Y
            """;

    private final String INPUT_DONT_WANT_TO_CONTINUE_GAME = """
            N
            """;

    private final String OUTPUT_GAME_START = """
              Heyho, let's play TicTacToe!
              Please enter the name of the first player.
              Ok, you have the Symbol X
              Please enter the name of the second player.
              Ok, you have the Symbol O
            """;

    private final String OUPUT_PLAYER1_BEGINS_AND_WINS = """
            Let's start!
                1   2   3
             A    |   |
                -----------
             B    |   |
                -----------
             C    |   |
             {Player1}, please select a field e.g. A1
                1   2   3
             A  X |   |
                -----------
             B    |   |
                -----------
             C    |   |
             {Player2}, please select a field e.g. A1
                1   2   3
             A  X | O |
                -----------
             B    |   |
                -----------
             C    |   |
             {Player1}, please select a field e.g. A1
                1   2   3
             A  X | O |
                -----------
             B  X |   |
                -----------
             C    |   |
             {Player2}, please select a field e.g. A1
                1   2   3
             A  X | O |
                -----------
             B  X | O |
                -----------
             C    |   |
             {Player1}, please select a field e.g. A1
                1   2   3
             A  X | O |
                -----------
             B  X | O |
                -----------
             C  X |   |
             Wow, {Player1}, I think you are a professional, you won.
            """;

    private final String OUPUT_PLAYER2_BEGINS_AND_WINS = """
            Let's start!
                1   2   3
             A    |   |
                -----------
             B    |   |
                -----------
             C    |   |
             {Player2}, please select a field e.g. A1
                1   2   3
             A  O |   |
                -----------
             B    |   |
                -----------
             C    |   |
             {Player1}, please select a field e.g. A1
                1   2   3
             A  O | X |
                -----------
             B    |   |
                -----------
             C    |   |
             {Player2}, please select a field e.g. A1
                1   2   3
             A  O | X |
                -----------
             B  O |   |
                -----------
             C    |   |
             {Player1}, please select a field e.g. A1
                1   2   3
             A  O | X |
                -----------
             B  O | X |
                -----------
             C    |   |
             {Player2}, please select a field e.g. A1
                1   2   3
             A  O | X |
                -----------
             B  O | X |
                -----------
             C  O |   |
             Wow, {Player2}, I think you are a professional, you won.
            """;

    String OUTPUT_NO_PLAYER_WINS = """
            Let's start!
               1   2   3
            A    |   |
               -----------
            B    |   |
               -----------
            C    |   |
            {Player1}, please select a field e.g. A1
               1   2   3
            A  X |   |
               -----------
            B    |   |
               -----------
            C    |   |
            {Player2}, please select a field e.g. A1
               1   2   3
            A  X |   |
               -----------
            B    |   |
               -----------
            C  O |   |
            {Player1}, please select a field e.g. A1
               1   2   3
            A  X |   |
               -----------
            B  X |   |
               -----------
            C  O |   |
            {Player2}, please select a field e.g. A1
               1   2   3
            A  X | O |
               -----------
            B  X |   |
               -----------
            C  O |   |
            {Player1}, please select a field e.g. A1
               1   2   3
            A  X | O | X
               -----------
            B  X |   |
               -----------
            C  O |   |
            {Player2}, please select a field e.g. A1
               1   2   3
            A  X | O | X
               -----------
            B  X | O |
               -----------
            C  O |   |
            {Player1}, please select a field e.g. A1
               1   2   3
            A  X | O | X
               -----------
            B  X | O |
               -----------
            C  O | X |
            {Player2}, please select a field e.g. A1
               1   2   3
            A  X | O | X
               -----------
            B  X | O | O
               -----------
            C  O | X |
            {Player1}, please select a field e.g. A1
               1   2   3
            A  X | O | X
               -----------
            B  X | O | O
               -----------
            C  O | X | X
            Wow, you are both masters, you both won.
            """;

    private final String OUTPUT_NEW_ROUND_MESSAGE = """
            Do you want to play another round? [Y/N]
            """;


    private Store store;


    void setupTest(String input, String expectedOutput) {
        this.store = new Store(
                expectedOutput,
                new TestUserInterface(input)
        );
    }


    @Test
    void full_game_where_Player1_wins() {
        String input = INPUT_USER_NAMES + INPUT_PLAYER1_WINS + INPUT_DONT_WANT_TO_CONTINUE_GAME;
        String expectedOutput = OUTPUT_GAME_START + OUPUT_PLAYER1_BEGINS_AND_WINS + OUTPUT_NEW_ROUND_MESSAGE;

        validate(input, expectedOutput);
    }


    @Test
    void full_game_where_no_player_wins() {
        String input = INPUT_USER_NAMES + INPUT_NO_PLAYER_WINS + INPUT_DONT_WANT_TO_CONTINUE_GAME;
        String expectedOutput = OUTPUT_GAME_START + OUTPUT_NO_PLAYER_WINS + OUTPUT_NEW_ROUND_MESSAGE;

        validate(input, expectedOutput);
    }

    @Test
    void board_is_empty_after_new_round() {
        String input =
                INPUT_USER_NAMES
                        + INPUT_PLAYER1_WINS
                        + INPUT_WANT_TO_CONTINUE_GAME
                        + INPUT_PLAYER1_WINS
                        + INPUT_DONT_WANT_TO_CONTINUE_GAME;
        String output =
                OUTPUT_GAME_START
                        + OUPUT_PLAYER1_BEGINS_AND_WINS
                        + OUTPUT_NEW_ROUND_MESSAGE
                        + OUPUT_PLAYER2_BEGINS_AND_WINS
                        + OUTPUT_NEW_ROUND_MESSAGE;
        validate(input, output);
    }

    private void validate(String input, String expectedOutput) {
        setupTest(input, expectedOutput);

        gameIsPlayedWithInput();
        outputsAreComparedLineByLine();
        if (this.store.hasDeviations()) {
            showDeviations();
        }
    }

    private void showDeviations() {
        Assertions.fail("The following output did not match the expectations: \n" +
                this.store.getDeviations().stream()
                        .map(this::deviationToString)
                        .collect(Collectors.joining("\n")
                        ));
    }

    private String deviationToString(Deviation deviation) {
        return String.join("\n",
                "position:\t" + deviation.position,
                "expected:\t" + deviation.expected,
                "actual:\t\t" + deviation.actual,
                "____________");
    }

    private void outputsAreComparedLineByLine() {
        List<String> actualOutput = this.store.getTestUserInterface().getOutput();
        List<String> expectedOutput = Arrays.asList(this.store.getExpectedOutput().split("\n"));

        List<Deviation> deviations = new ArrayList<>();
        for (int position = 1; position <= Math.max(actualOutput.size(), expectedOutput.size()); position++) {
            String actual = stringAtPosition(actualOutput, position);
            String expected = stringAtPosition(expectedOutput, position);
            if (!actual.equals(expected)) {
                deviations.add(new Deviation(expected, actual, position));
            }
        }
        this.store.setDeviations(deviations);
    }

    private String stringAtPosition(List<String> output, int position) {
        // Attention: position starts counting at 1 and not at 0
        //            because that is the way that our domain experts count.
        //            Thus, position has to be transformed a 0-based index
        if (position <= output.size()) {
            int index = position - 1;
            return output.get(index).trim();
        }
        return "";
    }

    private void gameIsPlayedWithInput() {
        new TicTacToeGame(this.store.getTestUserInterface());
    }

    record Deviation(String expected, String actual, int position) {
    }

    public class Store {
        private final String expectedOutput;
        private final TestUserInterface testUserInterface;
        private List<Deviation> deviations;

        public Store(String expectedOutput, TestUserInterface testUserInterface) {
            this.expectedOutput = expectedOutput;
            this.testUserInterface = testUserInterface;
        }

        public String getExpectedOutput() {
            return expectedOutput;
        }

        public TestUserInterface getTestUserInterface() {
            return testUserInterface;
        }

        public void setDeviations(List<Deviation> deviations) {
            this.deviations = deviations;
        }

        public List<Deviation> getDeviations() {
            return this.deviations;
        }

        public boolean hasDeviations() {
            return !this.deviations.isEmpty();
        }
    }

}
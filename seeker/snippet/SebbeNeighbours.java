//date: 2022-10-04T17:30:32Z
//url: https://api.github.com/gists/fadd69f762bf37bfcadf820ea7f22f0d
//owner: https://api.github.com/users/Insanityandme

import java.util.Arrays;

public class SebbeNeighbor {
    public static void main(String[] args) {
        int[][] testData = {
                {1, 1, 2, 3, 3},
                {2, 1, 1, 2, 3},
                {3, 2, 2, 1, 2},
                {3, 3, 3, 3, 3}
        };

        String[][] result = getAllowedPosition(testData);
        printResult(result);
    }

    // Returns the value of an array item at position 'array[row][col]'
    // if the position is out of bounds it will return 0
    public static int arrayItemOrZero (int row, int col, int[][] input) {
        // this checks whether the row is out of bounds OR the actual cell
        if (row < 0 || col < 0) {
            return 0;
        }

        if (row >= input.length || col >= input[row].length) {
            return 0;
        }

        return input[row][col];
    }

    // Returns the total value of all adjacent values to 'array[row][col]'
    public static int calculatedAdjacentTotal(int row, int col, int input[][]) {
        // 3 values above the player
        int valuesAbove = 0;
        valuesAbove += arrayItemOrZero(row - 1, col - 1, input);
        valuesAbove += arrayItemOrZero(row - 1, col, input);
        valuesAbove += arrayItemOrZero(row - 1, col + 1 , input);

        // 3 values below the player
        int valubesBelow = 0;
        valubesBelow += arrayItemOrZero(row + 1, col - 1, input);
        valubesBelow += arrayItemOrZero(row + 1, col, input);
        valubesBelow += arrayItemOrZero(row + 1, col + 1, input);

        // value to the left of the player
        int valueLeft = arrayItemOrZero(row, col - 1, input);

        // value to the right of the player
        int valueRight = arrayItemOrZero(row, col + 1, input);

        return valuesAbove + valubesBelow + valueLeft + valueRight;
    }

    // Check whether the position 'array[row][col]' is allowed ("T") or not ("F")
    public static String isPositionAllowed(int row, int col, int[][] input) {
        int currentPosition = input[row][col];

        if (currentPosition >= 3) {
            return "F";
        }

        int adjacentTotal = calculatedAdjacentTotal(row, col, input);

        if (adjacentTotal >= 15) {
            return "F";
        }

        else return "T";
    }

    // Get allowed player positions as an 2d array of "T" (allowed) and "F"'s (not allowed)
    public static String[][] getAllowedPosition(int input[][]) {
        String[][] output = new String[4][5];

        for (int row = 0; row < input.length; row++) {
            for (int col = 0; col < input[row].length; col++) {
                // Set the current cell to T/F
                output[row][col] = isPositionAllowed(row, col, input);
            }
        }

        return output;
    }

    public static void printResult(String[][] result) {
        for (int i = 0; i < result.length; i++) {
            System.out.printf("Row %d %s%n", i+1, Arrays.toString(result[i]));
        }
    }
}
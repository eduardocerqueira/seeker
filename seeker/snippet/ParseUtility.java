//date: 2023-12-21T17:07:58Z
//url: https://api.github.com/gists/4e8ef2e97943fb4d96cebcf52ad876c2
//owner: https://api.github.com/users/Archonic944

import java.util.Arrays;
import java.util.Scanner;

public class ParseUtility {
    static String[] asArray(String line){
        return line.split(" ");
    }

    static int[] asIntArray(String line){
        return asIntArray(asArray(line));
    }

    static int[] asIntArray(String[] line){
        return Arrays.stream(line).mapToInt(Integer::parseInt).toArray();
    }

    Scanner scanner;
    public String[][] table;
    public ParseUtility(Scanner scanner){
        this.scanner = scanner;
    }

    /**
     * Takes in new data of the specified length. Must be called before other operations; previous data will be replaced.
     * @param length amount of rows to take from the scanner
     */
    public void readTable(int length){
        table = new String[length][];
        for(int i = 0; i<length; i++){
            table[i] = scanner.nextLine().split(" ");
        }
    }

    public int integerAt(int row, int column){
        return Integer.parseInt(strAt(row, column));
    }

    public int[] intArrayAt(int row){
        return asIntArray(table[row]);
    }

    public String strAt(int row, int column){
        return table[row][column];
    }
}

//date: 2022-09-27T17:12:07Z
//url: https://api.github.com/gists/d3ee263cd65d9448bc775dac235a7721
//owner: https://api.github.com/users/Hilyas68

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Design and write a Java program that:
 * <p>
 * 1.	reads in a list of words from an array
 * 2.	determines the number of words in the list that start with each letter of the alphabet
 * 3.	outputs the results to the console, in alphabetical order (one letter per line)
 * <p>
 * Example Input:
 * String[] fruits = {"Apple", "Apricot", "Cherry", "Banana", "Cherry", "Papaya", "Cantaloupe"};
 * <p>
 * Example Output:
 * <p>
 * A: 2
 * B: 1
 * C: 3
 * D: 0
 * E: 0
 * F: 0
 * G: 0
 * H: 0
 * I: 0
 * J: 0
 * K: 0
 * L: 0
 * M: 0
 * N: 0
 * O: 0
 * P: 1
 * Q: 0
 * R: 0
 * S: 0
 * T: 0
 * U: 0
 * V: 0
 * W: 0
 * X: 0
 * Y: 0
 * Z: 0
 * <p>
 * 4.Add this output as a choice when the program starts. The new output should be like this:
 * <p>
 * A: 1
 * 1 Apple
 * <p>
 * B: 2
 * 1 Banana
 * 1 Blueberry
 * <p>
 * C: 1
 * 1 Cherry
 * <p>
 * D:  0
 * <p>
 * etc...
 */

public class WordThatStartWithAlphabetCount {


    static void wordThatStartWithAlphabetCount(String[] arr) {

        printTemplateOutput();

        Map<Character, Integer> alphabetCountMap = constructAlphabets();

        for (int i = 0; i < arr.length; i++) {
            if (arr[i] != null && !arr[i].isEmpty()) {
                char firstChar = arr[i].charAt(0);
                if (alphabetCountMap.containsKey(firstChar)) {
                    alphabetCountMap.put(firstChar, alphabetCountMap.get(firstChar) + 1);
                }
            }
        }

        //print to console
        alphabetCountMap.entrySet().forEach(alphabet -> {
            System.out.println(alphabet.getKey() + ":" + alphabet.getValue());
        });
    }

    //build the Alphabets
    static Map<Character, Integer> constructAlphabets() {
        Map<Character, Integer> alphabetCountMap = new LinkedHashMap<>();
        alphabetCountMap.put('A', 0);
        alphabetCountMap.put('B', 0);
        alphabetCountMap.put('C', 0);
        alphabetCountMap.put('D', 0);
        alphabetCountMap.put('E', 0);
        alphabetCountMap.put('F', 0);
        alphabetCountMap.put('G', 0);
        alphabetCountMap.put('H', 0);
        alphabetCountMap.put('I', 0);
        alphabetCountMap.put('J', 0);
        alphabetCountMap.put('K', 0);
        alphabetCountMap.put('L', 0);
        alphabetCountMap.put('M', 0);
        alphabetCountMap.put('N', 0);
        alphabetCountMap.put('O', 0);
        alphabetCountMap.put('P', 0);
        alphabetCountMap.put('Q', 0);
        alphabetCountMap.put('R', 0);
        alphabetCountMap.put('S', 0);
        alphabetCountMap.put('T', 0);
        alphabetCountMap.put('U', 0);
        alphabetCountMap.put('V', 0);
        alphabetCountMap.put('W', 0);
        alphabetCountMap.put('X', 0);
        alphabetCountMap.put('Y', 0);
        alphabetCountMap.put('Z', 0);

        return alphabetCountMap;
    }

    static void printTemplateOutput() {
        System.out.println("************ START Sample Output*************");
        System.out.println("A: 1");
        System.out.println("1 Apple");
        System.out.println();
        System.out.println("B: 2");
        System.out.println("1 Banana");
        System.out.println("1 Blueberry");
        System.out.println();
        System.out.println("C: 1");
        System.out.println("1 Cherry");
        System.out.println();
        System.out.println("D:  0");
        System.out.println();
        System.out.println("etc...");
        System.out.println("************ END Sample Output*************");
        System.out.println();
    }

    public static void main(String[] args) {

        String[] fruits = {null, "+Appl", "Apricot", "Cherry", "Banana", "Cherry", "Papaya", "Cantaloupe"};

        wordThatStartWithAlphabetCount(fruits);
    }
}

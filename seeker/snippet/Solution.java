//date: 2023-08-08T17:06:45Z
//url: https://api.github.com/gists/5895b0c8d08b13fa7b61b5c7804550ba
//owner: https://api.github.com/users/diegoehg

import java.io.*;
import java.math.*;
import java.security.*;
import java.text.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.function.*;
import java.util.regex.*;
import java.util.stream.*;
import static java.util.stream.Collectors.joining;
import static java.util.stream.Collectors.toList;

class Result {

    /*
     * Complete the 'migratoryBirds' function below.
     *
     * The function is expected to return an INTEGER.
     * The function accepts INTEGER_ARRAY arr as parameter.
     */

    public static int migratoryBirds(List<Integer> arr) {
        int[] sightingsCount = {0, 0, 0, 0, 0, 0};

        for (Integer i:arr)
            sightingsCount[i]++;
        
        int mostSightedType = 0;
        int maximumSightingsCount = 0;
        for (int i = 1; i <= 5; i++)
            if (sightingsCount[i] > maximumSightingsCount) {
                mostSightedType = i;
                maximumSightingsCount = sightingsCount[i];
            }
        
        return mostSightedType;
    }

}

public class Solution {
    public static void main(String[] args) throws IOException {
        BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(System.in));
        BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(System.getenv("OUTPUT_PATH")));

        int arrCount = Integer.parseInt(bufferedReader.readLine().trim());

        List<Integer> arr = Stream.of(bufferedReader.readLine().replaceAll("\\s+$", "").split(" "))
            .map(Integer::parseInt)
            .collect(toList());

        int result = Result.migratoryBirds(arr);

        bufferedWriter.write(String.valueOf(result));
        bufferedWriter.newLine();

        bufferedReader.close();
        bufferedWriter.close();
    }
}

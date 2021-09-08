//date: 2021-09-08T17:06:40Z
//url: https://api.github.com/gists/02cadebdf6c265884485ee13a7e3e74f
//owner: https://api.github.com/users/so77id

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
     * Complete the 'findAllMirrorNumbers' function below.
     *
     * The function is expected to return a STRING_ARRAY.
     * The function accepts INTEGER n as parameter.
     */
    
    public static ArrayList<String> helper(int n, int j, Map<Integer, Integer> comb, ArrayList<Integer> pos, ArrayList<Integer> baseCase) {
        ArrayList<String> sol = new ArrayList<String>();
        if (n == 0)
            return new ArrayList<String>(
                Arrays.asList("")
            );
        if (n == 1){
            for(int bc: baseCase) {
                sol.add(String.valueOf(bc));
            }
        }
        else {
            ArrayList<String> ret = helper(n-2, j, comb, pos, baseCase);
            for(int i = 0; i < 5; i++) {
                for(String r: ret) {
                    if(j == n && i == 0) continue;
                    sol.add(String.valueOf(pos.get(i)) + r + String.valueOf(comb.get(pos.get(i))));
                }    
            }
            
        }
        
        return sol;
    }
    
    public static ArrayList<String> findAllMirrorNumbers(int n) {
        Map<Integer, Integer> comb = new HashMap<Integer, Integer>() {{
                put(0, 0);
                put(1, 1);
                put(6, 9);
                put(8, 8);
                put(9, 6);
        }};
        ArrayList<Integer> pos = new ArrayList(
            Arrays.asList(0,1,6,8,9)
        );
        ArrayList<Integer> baseCase = new ArrayList(
            Arrays.asList(0,1,8)
        );


        return helper(n, n, comb, pos, baseCase);
    }

}

public class Solution {
    public static void main(String[] args) throws IOException {
        BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(System.in));
		BufferedWriter bufferedWriter = new BufferedWriter(new OutputStreamWriter(System.out));

        int n = Integer.parseInt(bufferedReader.readLine().trim());

        ArrayList<String> res = Result.findAllMirrorNumbers(n);
		
		Collections.sort(res);
      
      
        bufferedWriter.write(
            res.stream()
                .collect(joining(" "))
            + "\n"
        );

        bufferedReader.close();
        bufferedWriter.close();
    }
}
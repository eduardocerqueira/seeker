//date: 2022-09-12T16:56:33Z
//url: https://api.github.com/gists/a1f5f221ca36fb240851663842318e16
//owner: https://api.github.com/users/jpalvesloiola

import java.io.*;
import java.util.*;



public class Solution {
    public static void main(String[] args) throws IOException {
        BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(System.in));

        List<List<Integer>> arr = new ArrayList<>();

        for (int i = 0; i < 6; i++) {
            String[] arrRowTempItems = bufferedReader.readLine().replaceAll("\\s+$", "").split(" ");

            List<Integer> arrRowItems = new ArrayList<>();

            for (int j = 0; j < 6; j++) {
                int arrItem = Integer.parseInt(arrRowTempItems[j]);
                arrRowItems.add(arrItem);
            }

            arr.add(arrRowItems);
        }

        bufferedReader.close();
        
        Solution.largestSumHourglasses(arr);
        
    }
    
    static void largestSumHourglasses(List<List<Integer>> arr){
        int sum = Integer.MIN_VALUE;
        
        for (int i = 0; i < arr.size()-2; i++) {
            for (int j = 0; j < arr.size()-2; j++) {
                int temp = arr.get(i).get(j) + arr.get(i).get(j+1) + arr.get(i).get(j+2);
                temp += arr.get(i+1).get(j+1);
                temp += arr.get(i+2).get(j) + arr.get(i+2).get(j+1) + arr.get(i+2).get(j+2);
                if (sum < temp) {
                   sum = temp; 
                }
            }
        }
        System.out.println(sum);
    }
}
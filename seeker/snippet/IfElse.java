//date: 2025-09-11T16:55:00Z
//url: https://api.github.com/gists/45a1e03d34463d61bcbc8c89d8695c04
//owner: https://api.github.com/users/Pratiksha02-hub

import java.io.*;
import java.math.*;
import java.security.*;
import java.text.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.regex.*;

public class Solution {



    private static final Scanner scanner = new Scanner(System.in);

    public static void main(String[] args) {
        int N = scanner.nextInt();
        scanner.skip("(\r\n|[\n\r\u2028\u2029\u0085])?");
        
        if(N%2!=0) {
            System.out.println("Weird");
        }
        
        else{
            if(N>=2 && N<=5 && N%2 == 0) {
                System.out.println("Not Weird");
            }
            else if (N>=6 && N<=20 && N%2 == 0){
                System.out.println("Weird");
            }
            else if (N%2 == 0 && N>20){
                System.out.println("Not Weird");
            }
        }

        scanner.close();
    }
}

//date: 2022-09-21T17:22:46Z
//url: https://api.github.com/gists/bf180411127b2af35929ad5153551650
//owner: https://api.github.com/users/Sara-Pak

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



public class ConditionalStatements {
    public static void main(String[] args) throws IOException {
        BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(System.in));

        int N = Integer.parseInt(bufferedReader.readLine().trim());

        //first condition : If N is odd, print Weird
        if (N%2 !=0)
        {
            System.out.println("Weird");
        }
        else
        //second condition : If N is even and in the inclusive range of 2 to 5 , print Not Weird
        if(N>=2 && N<=5)
        {
            System.out.println("Not Weird");
        }
        //third condition : If N is even and in the inclusive range of 6 to 20, print Weird
        else if (N>=6 && N<=20)
        {
            System.out.println("Weird");
        }
        //fourth condition : If N is even and greater than 20, print Not Weird
        else if (N>20)
        {
            System.out.println("Not Weird");
        }
        bufferedReader.close();
    }
}

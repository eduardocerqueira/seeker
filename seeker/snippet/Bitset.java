//date: 2025-11-04T17:05:12Z
//url: https://api.github.com/gists/7c4b374408411557a22bbf33f8677818
//owner: https://api.github.com/users/Pratiksha02-hub

import java.io.*;
import java.util.*;
import java.text.*;
import java.math.*;
import java.util.regex.*;

public class Bitset {

    public static void main(String[] args) {
        /* Enter your code here. Read input from STDIN. Print output to STDOUT. Your class should be named Solution. */
        Scanner sc = new Scanner(System.in);
        int N = sc.nextInt(); // size of BitSets
        int M = sc.nextInt(); // number of operations
        
        BitSet[] bitsets = new BitSet[3];
        bitsets[1] = new BitSet(N);
        bitsets[2] = new BitSet(N);
        
        for (int i = 0; i < M; i++) {
            String operation = sc.next();
            int x = sc.nextInt();
            int y = sc.nextInt();
            
            switch (operation) {
                case "AND":
                    bitsets[x].and(bitsets[y]);
                    break;
                case "OR":
                    bitsets[x].or(bitsets[y]);
                    break;
                case "XOR":
                    bitsets[x].xor(bitsets[y]);
                    break;
                case "FLIP":
                    bitsets[x].flip(y);
                    break;
                case "SET":
                    bitsets[x].set(y);
                    break;
            }
            System.out.println(bitsets[1].cardinality() + " " + bitsets[2].cardinality());
        }
        sc.close();
    }
}
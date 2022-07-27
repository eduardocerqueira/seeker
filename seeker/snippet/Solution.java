//date: 2022-07-27T17:22:10Z
//url: https://api.github.com/gists/cdbf2e5a95c73026c327c3ee5fcd2f00
//owner: https://api.github.com/users/jpalvesloiola

import java.util.Scanner;

public class Solution {

    static boolean isAnagram(String a, String b) {
        // Complete the function
        String aSorted = sort(a.toUpperCase());
        String bSorted = sort(b.toUpperCase());
        
        return aSorted.equals(bSorted);
    }
    static String sort(String string){
        char[] charSort = string.toCharArray();
        java.util.Arrays.sort(charSort);
        return new String(charSort);
    }
    

  public static void main(String[] args) {
    
        Scanner scan = new Scanner(System.in);
        String a = scan.next();
        String b = scan.next();
        scan.close();
        boolean ret = isAnagram(a, b);
        System.out.println( (ret) ? "Anagrams" : "Not Anagrams" );
    }
}

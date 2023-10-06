//date: 2023-10-06T17:06:57Z
//url: https://api.github.com/gists/00572678363022bf78b12dca0defa879
//owner: https://api.github.com/users/theaman05

/** Get the reverse of a number */

import java.util.Scanner;

public class ReverseNumber{

    public static void main(String[] args){

        int num;
        Scanner sc = new Scanner(System.in);

        System.out.print("Enter a number: ");
        num = sc.nextInt();

        System.out.println(getReverseOf(num));

    }

    static int getReverseOf(int n){
        int res = 0, rem;

        while(n!=0){
            rem = n%10;
            n = n/10;
            res = res*10+rem;
        }
        
        return res;
    }

}
//date: 2023-10-06T17:08:05Z
//url: https://api.github.com/gists/15ed6693926c9341902f73d02e66974c
//owner: https://api.github.com/users/theaman05

/** Check whether an integer is a power of two or not using bitwise operators */

import java.util.Scanner;

public class BitwisePowerTwo{

    public static void main(String[] args){

        int num;
        Scanner sc = new Scanner(System.in);

        System.out.print("Enter a number: ");
        num = sc.nextInt();

        if(isPowerOfTwo(num)){
            System.out.println(num+" is "+"power of two");
        }
        else{
            System.out.println(num+" is "+"not power of two");
        }

    }

    static boolean isPowerOfTwo(int num){
        if((num&(num-1))==0){
            return true;
        }
        else{
            return false;
        }
    }

}
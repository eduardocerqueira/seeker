//date: 2021-11-12T17:14:55Z
//url: https://api.github.com/gists/5d41bfa346ea9700156e319f3e4a670b
//owner: https://api.github.com/users/FollowerOfBigboss

import java.util.Scanner;

public class HelloWorld{

     public static void main(String []args){
         Scanner sc = new Scanner(System.in);
         try
         {
             double input = sc.nextDouble();
             System.out.println("Your input is " + Double.toString(input)); 
         }
         catch(Exception e)
         {
            System.out.println("Please enter double!");         
         }

     }
}
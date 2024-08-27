//date: 2024-08-27T16:51:43Z
//url: https://api.github.com/gists/91aae8961d6274e074562132fc06edc0
//owner: https://api.github.com/users/sasub-mlp

import java.util.Scanner;

public class thirty_seven {
    public static void main(String[] args){
        Scanner scanner=new Scanner(System.in);
        int fact=1;
        System.out.println("Enter the number to find the factorial of it: ");
        int n= scanner.nextInt();
        int m=n;
        while(n!=0){
            fact*=n;
            n--;
        }
        System.out.println("Factorial of "+m+" is : "+fact);
    }
}

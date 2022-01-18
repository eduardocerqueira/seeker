//date: 2022-01-18T17:19:00Z
//url: https://api.github.com/gists/345a1ea6c3e06aeeba5dfb4a42947318
//owner: https://api.github.com/users/Arjun2002tiwari

import java.util.Scanner;

public class Factorial{
    public static void main(String[] args) {
        
        int i=10,num,fact=1,count=0;
        Scanner sc=new Scanner(System.in);
        num=sc.nextInt();
        int n=num;
        while(num>0){
            fact=fact*num;
            num--;
        }
        while(fact%i==0){
            count++;
            i=i*10;
        }
        System.out.println("Number of zeros in factorial of "+n+" is "+count);
        




    }
    
}

//date: 2022-10-06T17:09:04Z
//url: https://api.github.com/gists/ce8d399c763a3824f6d6e862993e8365
//owner: https://api.github.com/users/LisTata

package com.company;

import java.util.Scanner;

public class Ternary {
    public static void main(String[] args) {
        //Продажа табачных изделий до 18 лет запрещена
        int age= 18;
        System.out.println((age>=18 ? "Yes": "No"));
        //Строчная или заглавная
        System.out.printf("%d, %d, %d, %d, %n ", (int)'a', (int)'z',(int)'A', (int)'Z');
        char ch='b';
        String answer = (int)ch>=97 && (int)ch<=122 ? "lowercase":(int)ch>=65 &&(int)ch<=90?"uppercase":"not a letter";
        System.out.println(answer);

        //Наибольшее из 3 чисел.
        int a=1, b=3, c=2;
        //int max = a>b? b>c? a:a>c? a:c:a>c? b: b>c? b:c;
        int max=a>b && a>c? a:
                b>a && b>c? a:
                c>a && c>b? c: 0;
        int a2 = (int)(Math.random()*100+1);
       // System.out.println(a2);
        int b2=0;
        Scanner sc = new Scanner(System.in);

        while (a2!=b2){
            System.out.println("Enter number: ");
             b2 =sc.nextInt();
             if(a2>b2)
                 System.out.println("больше");
             else
                 System.out.println("меньше");

        }
        System.out.println("Win");







    }
}

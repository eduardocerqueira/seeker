//date: 2023-06-20T16:33:18Z
//url: https://api.github.com/gists/9e304bd724ceecce31f63ca42e22cd71
//owner: https://api.github.com/users/AytanShabanova

package org.example;

import java.util.Scanner;

public class Task1 {
    public static void main(String[] args) {
        Scanner sc=new Scanner(System.in);
        System.out.println("Enter your Fizik result");
        int fizik= sc.nextInt();
        System.out.println("Enter your Kimya result");
        int kimya=sc.nextInt();
        System.out.println("Enter your Turkce result");
        int turkce=sc.nextInt();
        System.out.println("Enter your Tarih result");
        int tarih=sc.nextInt();
        System.out.println("Enter your Muzik result");
        int muzik=sc.nextInt();
        int sum=(fizik+kimya+turkce+tarih+muzik)/5;
        String result=(sum>=60)?"Sinavi gecdi":"Sinavi gecemedi";
        System.out.println(result);
    }
}

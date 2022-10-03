//date: 2022-10-03T17:20:06Z
//url: https://api.github.com/gists/fd4f6ed5f7205b630516595ceee37423
//owner: https://api.github.com/users/ebede2g

package com.jtp.modSUS;

import com.sun.jdi.IntegerValue;

import java.util.ArrayList;
import java.util.Scanner;

public class lab4 {
    public static void main(String[] args) {
        Scanner scan = new Scanner(System.in);

        System.out.println("Введіть число : ");
        double xFULL = scan.nextDouble();
        int xINT = (int)xFULL;                          // Ціла частина числа
        double xDROB = xFULL-xINT;                      // Дробова частина

        int t;

        if(xDROB==0) {                                  // Перевірка а цілочисельність
            System.out.println("число є цілим");
            ArrayList<Integer> Arr = new ArrayList<>(); // Масив з майбутніми елементами
            t = xINT;                                   // Дублюю число


            for(;t>=1;t/=2)
                Arr.add(t % 2);
            for(int gen = Arr.size()-1; gen >= 0 ; gen--)
                System.out.print(Arr.get(gen));






            }





        else{
            System.out.println("число є дробовим");

        }













    }
}

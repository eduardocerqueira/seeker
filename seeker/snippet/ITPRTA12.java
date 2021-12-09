//date: 2021-12-09T17:11:39Z
//url: https://api.github.com/gists/091ad2a997d882bab5a323618ddad968
//owner: https://api.github.com/users/mochammadrafi

import java.util.Scanner;
import java.lang.Math;

public class ITPRTA12 {
    public static void main(String[] args) {
        double a, b, c, answerA, answerB;
        
        Scanner input = new Scanner(System.in);
        System.out.print("Input number 1:");
        a = input.nextDouble();
        System.out.print("Input number 2:");
        b = input.nextDouble();
        System.out.print("Input number 3:");
        c = input.nextDouble();

        answerA = Math.log10(a) + Math.log10(b) + Math.log10(c);
        answerB = Math.cos(a) - Math.sin(b) - Math.tan(c);

        System.out.println("Answer A is :" + answerA);
        System.out.println("Answer B is :" + answerB);
    }
}
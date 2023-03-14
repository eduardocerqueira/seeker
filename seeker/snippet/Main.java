//date: 2023-03-14T16:58:25Z
//url: https://api.github.com/gists/e010170d6f68b6fde198ac1ec9944bac
//owner: https://api.github.com/users/Drogunov-S

package ru.drogunov.springproject;

import java.util.Scanner;

public class Main {
    private static int dano = 7;
    /*
     * На любом языке программирования написать алгоритм, который для любых
     * вещественных a, b и с
     * решает уравнение: а*x^2+b*x+c = 7
     * */
    
    public static void main(String[] args) {
        Main main = new Main();
        Scanner scanner = new Scanner(System.in);
        double a = scanner.nextDouble();
        double b = scanner.nextDouble();
        double c = scanner.nextDouble();
        main.result(a, b, c, dano);
        scanner.close();
    }
    
    public void result(double a, double b, double c, double number) {
        double disc = Math.pow(b, 2) - (4 * a * (c - number));
        System.out.println(disc);
        if (disc > 0) {
            double x1 = a == 0 ? (-b + Math.sqrt(disc)) : (-b + Math.sqrt(disc)) / (2 * a);
            double x2 = a == 0 ? (-b - Math.sqrt(disc)) : (-b - Math.sqrt(disc)) / (2 * a);
            System.out.printf("Уравнение имеет два корня x1=%.2f x2=%.2f", x1, x2);
        } else if (disc == 0) {
            double x = a == 0 ? -b : -b / (2 * a);
            System.out.printf("корень уравнения равен x=%.2f", x);
        } else {
            System.out.println("Нужно вспомнить математику, а уравнение не имеет коренней");
        }
    }
}

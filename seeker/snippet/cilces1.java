//date: 2024-09-30T17:11:50Z
//url: https://api.github.com/gists/486952442b0036aee365e7e056ba2287
//owner: https://api.github.com/users/itpark-team

import java.util.Scanner;

public class Main {
    public static void main(String[] args) {

        Scanner scanner = new Scanner(System.in);

        double distance, increasePercent;
        int days;

        System.out.print("Введите начальную дистанцию(км): ");
        distance = scanner.nextDouble();

        System.out.print("Введите увеличение дистанции в процентах каждый день: ");
        increasePercent = scanner.nextDouble();

        System.out.print("Введите кол-во дней пробежек: ");
        days = scanner.nextInt();

        for (int i = 0; i < days; i++) {
            distance += distance * increasePercent / 100.0;
            
            System.out.println(String.format("бегун после %d дня пробежал %.2f км", i, distance));
        }
        
        
//        int i = 0;
//        while (i < days) {
//            //distance = distance +
//            distance += distance * increasePercent / 100.0;
//            i++;
//
//            System.out.println(String.format("бегун после %d дня пробежал %.2f км", i, distance));
//
//            //System.out.printf("бегун после %d дня пробежал %.2f км\n", i, distance);
//        }

    }
}
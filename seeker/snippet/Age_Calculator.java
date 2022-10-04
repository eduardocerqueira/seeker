//date: 2022-10-04T17:24:57Z
//url: https://api.github.com/gists/d69e622a716bbd5aa2542ae56bc34459
//owner: https://api.github.com/users/RedBeard2000

import java.util.Objects;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        System.out.println("What year were you born?");
        var birth_year = scanner.nextInt();

        System.out.println("What year is today?");
        var today_year = scanner.nextInt();

        var age = today_year - birth_year;
        if (age < 0)
            System.out.println("You haven't been born yet");
        else {
            System.out.println("Did you already have a birthday this year?");
            var ans = scanner.next();
            if (Objects.equals(ans, "No"))
                age = age - 1;
            if (age < 0)
                System.out.println("You haven't been born yet");
            if (age == 0)
                System.out.println("Congratulations! You were born! Welcome to the world!");
            if (age == 1)
                System.out.println("You are one year born");
            if (age > 1)
                System.out.println("You are " + age + " years old");
        }
    }
}
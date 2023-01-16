//date: 2023-01-16T16:42:59Z
//url: https://api.github.com/gists/922566d1ab353864c721f5ebc89c51a5
//owner: https://api.github.com/users/Mikaeryu

//Задание 5.3.13 https://stepik.org/lesson/12784/step/13?unit=3131

import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        double sum = 0;

        while(scanner.hasNext()) {
            if(scanner.hasNextDouble()) {
                    sum += Double.parseDouble(scanner.next());
            } else {
                scanner.next();
            }
        }

        System.out.format("%.6f", sum);
    }
}
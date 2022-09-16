//date: 2022-09-16T17:05:33Z
//url: https://api.github.com/gists/7549591be976ed4f02adca135bd20000
//owner: https://api.github.com/users/RusiArabadzhiev

package programming_basics_softuni;

import java.util.Scanner;

public class aquaruim {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        double length = Double.parseDouble(scanner.nextLine());
        double width = Double.parseDouble(scanner.nextLine());
        double height = Double.parseDouble(scanner.nextLine());
        double percent = Double.parseDouble(scanner.nextLine());


        double volumeSm = length * width * height;
        double volumeDm = volumeSm / 1000.00;
        double waterneed = volumeDm * (1 - (percent / 100));

        System.out.println(waterneed);
    }
}

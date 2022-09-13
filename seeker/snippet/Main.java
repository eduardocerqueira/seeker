//date: 2022-09-13T17:21:04Z
//url: https://api.github.com/gists/832d532c394d603d33e1c95431db08e1
//owner: https://api.github.com/users/hehetenya

import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        System.out.println("What is Bitcoin price today?");
        int bitcoinPrice = scanner.nextInt();

        System.out.println("How much $ do you have?");
        int dollarsAvailable = scanner.nextInt();

        double result = (double) dollarsAvailable / bitcoinPrice;

        System.out.printf("You can buy %f BTC%n", result);
    }
}
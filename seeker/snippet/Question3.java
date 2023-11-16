//date: 2023-11-16T17:00:37Z
//url: https://api.github.com/gists/9d0e7715c0ffaabcade3d77247f7f785
//owner: https://api.github.com/users/mahfuz-oxyfw

import java.util.Scanner;

public class Question3 {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        System.out.print("Enter capital to invest: ");
        double capital = scanner.nextDouble();
        final double interestRate = 12.5;

        for (int year = 1; year <= 6; year++) {
            double interest = capital * interestRate / 100;
            double newCapital = capital + interest;

            interest = Math.round(interest * 100.0) / 100.0;
            newCapital = Math.round(newCapital * 100.0) / 100.0;

            System.out.println("Year " + year);
            System.out.println("Interest = " + interest + " Cumulative interest = " + (Math.round((capital + interest) * 100.0) / 100.0) + " Capital = " + newCapital);

            capital = newCapital;
        }
    }
}

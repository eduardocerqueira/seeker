//date: 2022-12-02T17:06:00Z
//url: https://api.github.com/gists/ac2b36d4177bc02eaa5ad2ed5af99fe4
//owner: https://api.github.com/users/Alexsandor12

import java.time.temporal.ChronoUnit;
import java.util.Scanner;
import java.time.LocalDate;

public class Main {
    public static void main(String[] args) {
        Scanner c = new Scanner(System.in);
        System.out.println("Впишите дату первую: (Формат: YYYY.MM.DD)");
        String[] getting1;
        getting1 = c.next().split("\\.");
        int y1 = Integer.parseInt(getting1[0]);
        int m1 = Integer.parseInt(getting1[1]);
        int d1 = Integer.parseInt(getting1[2]);
        System.out.println("Впишите дату вторую: (Формат: YYYY.MM.DD)");
        String[] getting2;
        getting2 = c.next().split("\\.");
        int y2 = Integer.parseInt(getting2[0]);
        int m2 = Integer.parseInt(getting2[1]);
        int d2 = Integer.parseInt(getting2[2]);

        LocalDate date1 = LocalDate.of(y1, m1, d1);
        LocalDate date2 = LocalDate.of(y2, m2, d2);
        long days = ChronoUnit.DAYS.between(date1,date2);
        System.out.println("Между датами: " + days + " дней");

    }
}
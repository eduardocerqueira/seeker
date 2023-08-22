//date: 2023-08-22T16:52:04Z
//url: https://api.github.com/gists/a448224ab8c3d6ea5250774f10547665
//owner: https://api.github.com/users/Seungmi97

/*
황승미
제로베이스 백엔드 스쿨 16기
*/

import java.util.Scanner;

public class Main {
    public static int CalRate(int income, float rate, int standard) {
        int price = income;
        int tax;

        if (income > standard && standard != 0) {
            price = standard;
        }
        tax = (int)(price * rate);

        System.out.printf("%10d * %2d%% =\t%10d\n", price, (int)(rate * 100), tax);

        return tax;
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        int[][] rates = {{6, 12000000, 0}, {15, 46000000, 1080000}, {24, 88000000, 5220000}, {35, 150000000, 14900000}, {38, 300000000, 19400000}, {40, 500000000, 25400000}, {42, 1000000000, 35400000}, {45, 1000000000, 65400000}};

        int taxByRate = 0;
        int taxByPD = 0;

        try {
            System.out.println("[과세금액 계산 프로그램]");
            System.out.print("연소득을 입력해 주세요.:");
            int income = sc.nextInt();

            for (int[] rate : rates) {
                if (income <= rate[1] || rate[0] == 45) {
                    taxByPD = (int)(income * (float) rate[0] / 100) - rate[2];
                    break;
                }
            }

            for (int i = 0; i < rates.length; i++) {
                int standard = (i == 0) ? rates[i][1] : rates[i][1] - rates[i - 1][1];
                taxByRate += CalRate(income, (float) rates[i][0] / 100, standard);
                income -= standard;
                if (income <= 0) break;
            }

            System.out.println();
            System.out.println("[세율에 의한 세금]:\t\t\t\t" + taxByRate);
            System.out.println("[누진공제 계산에 의한 세금]:\t\t" + taxByPD);
        } catch(Exception e) {
            System.out.println("e = " + e);
        }
    }
}
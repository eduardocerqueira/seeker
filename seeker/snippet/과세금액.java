//date: 2022-04-07T16:51:07Z
//url: https://api.github.com/gists/007953ea6cb6b35b1b3bfebdfab5c3dd
//owner: https://api.github.com/users/AgFe2

import java.util.InputMismatchException;
import java.util.List;
import java.util.Scanner;

public class 과세금액 {
    public static void solution() {
        double inputInCome = 0;
        int totalTax = 0;
        int deducTax = 0;
        int taxSec[] = {6, 15, 24, 35, 38, 40, 42, 45};
        int deducTaxSec[] = {0, 108, 522, 1490, 1940, 2540, 3540, 6540};

        System.out.println("[과세금액 계산 프로그램]");


        //연소득 입력
        while (true) {
            try {
                System.out.print("연소득을 입력해 주세요.: ");
                Scanner sc = new Scanner(System.in);
                inputInCome = sc.nextDouble();
                break;
            } catch (InputMismatchException e) {
                System.out.println("숫자만 입력해주세요.");
            }
        }


        int myInCome[] = inComeCal(inputInCome);

        //세율에 의한 세금 계산
        for (int i = 0; i < 8; i++) {
            if (myInCome[i] > 0) {
                System.out.printf("%10d * %2d%% = %10d", myInCome[i], taxSec[i], (int)(myInCome[i] * 0.01 * taxSec[i]));
                System.out.println();
                totalTax += myInCome[i] * 0.01 * taxSec[i];
            }
        }

        //누진공제 계산에 의한 세금 계산
        for (int i = 0; i < 8; i++) {
            if (myInCome[i] == 0) {
                deducTax = (int)(inputInCome * 0.01 * taxSec[i - 1]) - (deducTaxSec[i - 1] * 10000);
                break;
            } else {
                deducTax = (int)(inputInCome * 0.01 * taxSec[7]) - (deducTaxSec[7] * 10000);
            }
        }

        System.out.println();
        System.out.printf("[세율에 의한 세금]:%19d", totalTax);
        System.out.println();
        System.out.printf("[누진공제 계산에 의한 세금]:　%10d", deducTax);
    }

    //소득 분할
    public static int[] inComeCal(double inCome) {
        int inComeSec[] = {12000000, 34000000, 42000000, 62000000, 150000000, 200000000, 500000000, 1000000000};
        int _inCome[] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int i = 0; i < 8; i++) {
            if (inCome / inComeSec[i] >= 1) {
                if (i == 7) {
                    _inCome[i] = (int)inCome;
                    break;
                }
                _inCome[i] = inComeSec[i];
                inCome -= inComeSec[i];
            } else {
                _inCome[i] = (int)inCome % inComeSec[i];
                inCome = 0;
            }
        }

        return _inCome;
    }

    public static void main(String[] arg) {
        solution();
    }
}
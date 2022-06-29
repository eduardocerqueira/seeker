//date: 2022-06-29T17:13:26Z
//url: https://api.github.com/gists/96e3631477917d1e3b90219902e8414a
//owner: https://api.github.com/users/Water-Stone

/*
    송주헌

    [1~2주차 미니과제]
    5. 달력 출력 프로그램 (배점 10점)
 */

import java.time.LocalDate;
import java.util.Scanner;

public class JavaQuiz05 {
    public static void main(String[] args) {
        while (true) {
            makeCalendar();
        }
    }

    public static void makeCalendar() {
        Scanner sc = new Scanner(System.in);

        System.out.println("[달력 출력 프로그램]");
        System.out.print("달력의 년도를 입력해주세요.(yyyy): ");
        int year = sc.nextInt();
        System.out.print("달력의 월을 입력해주세요.(mm): ");
        int month = sc.nextInt();

        LocalDate targetDate = LocalDate.of(year, month, 1);

        System.out.println(String.format("[%d년 %02d월]", year, month));
        System.out.println("일\t월\t화\t수\t목\t금\t토");

        int blanks = (targetDate.getDayOfWeek().getValue() % 7);    // 0~6: 일~토
        for (int i = 0; i < blanks; i++) {
            System.out.print("\t");
        }

        for (int days = 1; days <= targetDate.lengthOfMonth(); ) {
            if (days == 1) {
                for (int i = blanks; i < 7; i++) {
                    System.out.print(String.format("%02d\t", days));
                    days++;
                }
                System.out.println();
            } else {
                for (int i = 0; i < 7 && days <= targetDate.lengthOfMonth(); i++) {
                    System.out.print(String.format("%02d\t", days));
                    days++;
                }
                System.out.println();
            }
        }
    }
}

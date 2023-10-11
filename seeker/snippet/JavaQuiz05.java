//date: 2023-10-11T17:09:41Z
//url: https://api.github.com/gists/aaf8fc2911951741c8ba55c1934ec8b6
//owner: https://api.github.com/users/yje9802

// 양지은

import java.time.LocalDate;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        System.out.println("[달력 출력 프로그램]");
        // 스캐너 선언
        Scanner scanner = new Scanner(System.in);

        // 정보 입력 받기
        System.out.print("달력의 년도를 입력해 주세요. (yyyy): ");
        int year = scanner.nextInt();
        System.out.print("달력의 월을 입력해 주세요. (mm): ");
        int month = scanner.nextInt();

        // 로컬 데이트 선언
        LocalDate localDate = LocalDate.of(year, month, 1);
        // 해당 월의 1일이 무슨 요일인지
        // 월요일이면 0
        int firstDate = localDate.getDayOfWeek().getValue();
        int totalDays = localDate.lengthOfMonth();

        System.out.printf("\n[%d년 %02d월]\n", year, month);
        System.out.println("일\t월\t화\t수\t목\t금\t토");

        int week = 0; // 한 주의 몇 번째 날짜인지 체크용
        if (firstDate != 7) {
            for (int i = 0; i < firstDate; i++) {
                week += 1;
                System.out.print("\t");
            }
        }

        for (int day = 1; day < totalDays + 1; day++) {
            System.out.printf("%02d\t", day);
            week += 1;
            if (week % 7 == 0) {
                System.out.println();
            }
        }
    }
}
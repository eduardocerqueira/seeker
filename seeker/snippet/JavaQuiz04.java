//date: 2023-10-10T17:05:59Z
//url: https://api.github.com/gists/56e2290f3f6e93773daa79e6b4e6421b
//owner: https://api.github.com/users/yje9802

// 양지은

import java.util.Random;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        System.out.println("[주민등록번호 계산]");
        // 스캐너 선언
        Scanner scanner = new Scanner(System.in);
        // 랜덤 선언
        Random random = new Random();

        // 정보 입력 받기
        System.out.print("출생년도를 입력해 주세요. (yyyy): ");
        int year = scanner.nextInt();
        System.out.print("출생월을 입력해 주세요. (mm): ");
        int month = scanner.nextInt();
        System.out.print("출생일을 입력해 주세요. (dd): ");
        int day = scanner.nextInt();
        System.out.print("성별을 입력해 주세요. (m/f): ");
        String sex = scanner.next();

        int randomN = random.nextInt(1000000);
        int sexCode = 3;
        if (sex.equals("f")) {
            sexCode = 4;
        }

        System.out.printf("%d%02d%02d-%d%06d", year%2000, month, day, sexCode, randomN);
    }
}
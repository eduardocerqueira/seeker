//date: 2022-07-04T03:11:02Z
//url: https://api.github.com/gists/b90b63c590474f60ede07dd056949007
//owner: https://api.github.com/users/yj0903


// 채유진
// 미니프로젝트 Quiz 4

import java.util.Random;
import java.util.Scanner;

public class Quiz4 {
    public static void main(String[] args) {
        System.out.println("[주민등록번호 계산]");
        Scanner sc = new Scanner(System.in);

        Random random = new Random();



        System.out.print("출생년도를 입력해주세요.(yyyy): ");
        int year = Integer.parseInt(sc.nextLine());

        System.out.print("출생월을 입력해주세요.(mm): ");
        int month = Integer.parseInt(sc.nextLine());

        System.out.print("출생일을 입력해주세요.(dd): ");
        int day = Integer.parseInt(sc.nextLine());

        System.out.print("성별을 입력해주세요.(m/f): ");
        String s = sc.nextLine();


        int sex = 0;
        if (year >= 2000) {
            sex = s.equals("m") ? 3:4;

        } else {
            sex = s.equals("m") ? 1:2;
        }
        int other = random.nextInt(1000000);

        String registrationNumber = String.format("%02d%02d%02d-%d%06d", year%100, month, day, sex, other);
        System.out.println(registrationNumber);
    }
}

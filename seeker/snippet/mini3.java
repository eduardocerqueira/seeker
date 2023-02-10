//date: 2023-02-10T16:56:59Z
//url: https://api.github.com/gists/c3ec26525f4cbe4448d24b580c854e84
//owner: https://api.github.com/users/hwanghyerim

// 미니과제3
// 황혜림


import java.util.Scanner;

public class mini3 {
    public static void main(String[] args) {
        System.out.println("[입장권 계산]");
        int ticket = 0;
        Scanner sc = new Scanner(System.in);

        System.out.print("나이를 입력해 주세요.(숫자):");
        int age = sc.nextInt();
        System.out.print("입장시간을 입력해 주세요.(숫자입력):");
        int clock = sc.nextInt();
        System.out.print("국가유공자 여부를 입력해 주세요.(y/n):");
        char nationalMerit = sc.next().trim().charAt(0);
        System.out.print("복지카드 여부를 입력해 주세요.(y/n):");
        char welfareCard = sc.next().trim().charAt(0);

        if (age < 3) {
            ticket = 0;
        } else if (age >= 3 && age < 13) {
            ticket = 4000;
        } else if (clock >= 17) {
            ticket = 4000;
        } else if (nationalMerit == 'y' || welfareCard == 'y') {
            ticket = 8000;
        } else {
            ticket = 10000;
        }
        System.out.print("입장료: " + ticket);
    }
}
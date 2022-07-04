//date: 2022-07-04T03:10:26Z
//url: https://api.github.com/gists/2c61cd8db60c554cd594ffe698fb599a
//owner: https://api.github.com/users/yj0903


// 채유진
// 미니프로젝트 Quiz 3

import java.util.Scanner;

public class Quiz3 {

    public static void main(String[] args) {

        System.out.println("[입장권 계산]");
        Scanner sc = new Scanner(System.in);

        System.out.print("나이를 입력해주세요. (숫자):");
        int age = Integer.parseInt(sc.nextLine());

        System.out.print("입장시간을 입력해주세요. (숫자):");
        int time = Integer.parseInt(sc.nextLine());

        System.out.print("국가유공자 여부를 입력해주세요. (y/n):");
        String nationalMerit = sc.nextLine();

        System.out.print("복지카드 여부를 입력해주세요. (y/n):");
        String welfareCard = sc.nextLine();

        int fare;

        if (age < 3) {
            fare = 0;
        } else if (nationalMerit.equals('y') || welfareCard.equals('y')) {
            fare = 8000;
        } else if (age < 13 || time >= 17) {
            fare = 4000;
        } else {
            fare = 8000;
        }
        System.out.printf("입장료: %d", fare);
    }
}

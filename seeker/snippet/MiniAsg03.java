//date: 2023-08-01T17:03:17Z
//url: https://api.github.com/gists/75cba17d4c192cdeaea6babc11fdb9b8
//owner: https://api.github.com/users/Jbiscode

/*
사재빈

문제를 구현하면서 원하는 형식이 아니면 다시 입력을 받을 수 있는 예외처리까지 해보고싶었는데 잘 되지않았다.
구글링을 하면서라도 구현해보고싶었지만 실패했다.
토끼굴에 빠질것같기때문에 일단은 넘어가고 할수있는것에 집중해보자.
*/

import java.util.Scanner;

public class MiniAsg03 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("나이를 입력해 주세요.(숫자):");
        int age = sc.nextInt();
        System.out.print("입장시간을 입력해 주세요.(숫자입력):");
        int time = sc.nextInt();
        System.out.print("국가유공자 여부를 입력해 주세요.(y/n):");
        String hero = sc.next().toLowerCase();
        System.out.print("복지카드 여부를 입력해 주세요.(y/n):");
        String card = sc.next().toLowerCase();

        int cost = 10000;

        if (age < 3) {
            cost = 0;
        } else if (age < 13) {
            cost = 4000;
        } else if (time >= 17) {
            cost = 4000;
        } else if ((hero.equals("y") || card.equals("y"))) {
            cost = 8000;
        }

        if (cost == 0) {
            System.out.println("입장료: 무료입장");
        } else {
            System.out.printf("입장료: %d", cost);
        }
    }
}
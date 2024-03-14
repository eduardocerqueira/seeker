//date: 2024-03-14T16:57:17Z
//url: https://api.github.com/gists/2780de331ca7e18a64ef957b5c922c8f
//owner: https://api.github.com/users/jjm159

package org.example.assignment_03_18;

/* 정재명
 *
 * 놀이동산 입장권 계산 프로그램
 *
 * Scanner의 입력함수, 다중 조건문
 *
 */

import java.util.Scanner;
public class p03 {
    public static void main(String[] args) {

        Scanner scanner = new Scanner(System.in);

        System.out.println("[입장권 계산]");

        System.out.print("나이를 입력해 주세요. (숫자):");
        int age = scanner.nextInt();
        scanner.nextLine();

        System.out.print("입장시간을 입력해 주세요.(숫자입력):");
        int time = scanner.nextInt();
        scanner.nextLine();

        System.out.print("국가유공자 여부를 입력해 주세요.(y/n):");
        String isHero = scanner.next();

        System.out.print("복지카드 여부를 입력해 주세요.(y/n):");
        String hasWelfareCard = scanner.next();

        int fee = 10_000;

        int free = 0;
        int specialPrice = 4_000;
        int normalPrice = 8_000;

        String yes = "y";
        String no = "n";

        if (age < 3) {
            fee = free;
        }

        if (age < 13 || time > 17) {
            fee = Math.min(fee, specialPrice);
        }

        if (isHero.equals(yes) || hasWelfareCard.equals(yes)) {
            fee = Math.min(fee, normalPrice);
        }

        System.out.printf("입장료: %d%n", fee);
    }
}

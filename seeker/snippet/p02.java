//date: 2024-03-14T16:55:34Z
//url: https://api.github.com/gists/381ca599769585f8bf0fa484219d3833
//owner: https://api.github.com/users/jjm159

package org.example.assignment_03_18;

/* 정재명
 *
 * 결제 금액 캐시백 계산 프로그램
 * [캐시백 계산 조건]
 * - 결재 금액의 10%를 적립한다.
 * - 캐시백포인트 단위는 백원단위이다.(100원, 200원, 300원 등)
 * - 한건의 캐시백 포인트는 최대 300원을 넘을 수 없습니다.
 */

import java.util.Scanner;

public class p02 {
    public static void main(String[] args) {
        Scanner inScanner = new Scanner(System.in);

        System.out.print("[캐시백 계산]\n결제 금액을 입력해주세요.(금액):");

        int payAmount = inScanner.nextInt();
        int cashback = Math.max(payAmount / 10, 300); // 10% 적립, 최대 300

        String result = String.format("결제 금액은 %d원이고, 캐시백은 %d원 입니다.\n", payAmount, cashback);
        System.out.println(result);
    }
}

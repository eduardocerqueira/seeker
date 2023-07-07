//date: 2023-07-07T16:44:41Z
//url: https://api.github.com/gists/de90be38fdcbc20759b99ac74733bb0d
//owner: https://api.github.com/users/rorrorome

/*
 * 제로베이스 백엔드 스쿨 15기
 * 한새롬
 * 미니과제 8. 연소득 과세금액 계산 프로그램
 * 
 * Scanner의 입력함수와 조건문 및 반복문 과 배열, 함수를 통한 과세 로직 작성
 * 1. 연소득 금액 입력
 * 2. 주어진 종합소득세율표를 통한 조건 생성
 */

import java.util.Scanner;
import java.util.Arrays;

public class MiniProject0802 {

	public static void main(String[] args) {
		Scanner sc = new Scanner(System.in);
		System.out.println("[과세금액 계산 프로그램]");
		System.out.print("연소득을 입력해 주세요.:");
		int income = sc.nextInt();

		// 주어진 조건
		double[] rate = { 0.06, 0.15, 0.24, 0.35, 0.38, 0.40, 0.42, 0.45 }; // 비율(세율)
		int[] ptt = { 0, 12000000, 46000000, 88000000, 150000000, 300000000, 500000000, 1000000000 }; // 구분값(partition)
		int[] minus = { 0, 1080000, 5220000, 14900000, 19400000, 25400000, 35400000, 65400000 }; // 누진공제

		// 소득분위(?) 판단
		int part = 0;
		for (int i = 0; i < ptt.length; i++) {
			if (income > ptt[i]) {
				part = i;
			}
		}

		int[] st = new int[7]; // 구간 값(section)
		int[] pt = new int[7]; // 구간별 세금
		for (int j = 0; j < 7; j++) {
			st[j] = ptt[j + 1] - ptt[j];
			pt[j] = (int) (st[j] * rate[j]);
		}

		int v = income - ptt[part];
		int tax = (int) (v * rate[part]);
		int totalTax = tax;

		if (part != 0) {
			for (int k = 0; k < part; k++) {
				System.out.printf("%10d *%3d%% =%10d\n", st[k], (int) (rate[k] * 100), pt[k]);
				totalTax = totalTax + pt[k];
			}
			System.out.printf("%10d *%3d%% =%10d\n", v, (int) (rate[part] * 100), tax);
		} else {
			System.out.printf("%10d *%3d%% =%10d\n", v, (int) (rate[0] * 100), tax);
		}
		System.out.printf("\n[세율에 의한 세금]:\t\t\t%10d", totalTax);

		int mt = (int) (income * rate[part]) - minus[part]; // 누진공제 세금

		System.out.printf("\n[누진공제 계산에 의한 세금]:\t%10d", mt);
	}
}

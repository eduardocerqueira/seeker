//date: 2023-08-03T17:09:45Z
//url: https://api.github.com/gists/5f6c7bf5f5303dcd0bb67fda64d60b12
//owner: https://api.github.com/users/park-sang-yong

/*
  박상용
  1. Scanner 사용
  조건 확인하여 포인트를 설정 및 출력
  2. C언어의 printf가 익숙하여 format메소드 이용
*/

import java.util.Scanner;

public class mini_Sol2 {

	public static void main(String[] args) {
		Scanner sc = new Scanner(System.in);

		String[] str = new String[2];
		System.out.println("[캐시백 계산]");
		str[0] = String.format("결제 금액을 입력해 주세요.(금액):");
		
		System.out.print(str[0]);
		
		int cash = sc.nextInt();
		int point = cash / 1000 * 100;
		
		if(point > 300)
			point = 300;
		str[1] = String.format("결제 금액은 %d원이고, 캐시백은 %d원 입니다.", cash, point);
		System.out.print(str[1]);
	}
}
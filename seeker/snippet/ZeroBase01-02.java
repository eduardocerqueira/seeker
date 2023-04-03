//date: 2023-04-03T17:00:19Z
//url: https://api.github.com/gists/0009e0ea553a21d4729c3331072c98d9
//owner: https://api.github.com/users/Eun-SeoKim

import java.util.Scanner;

public class Main02 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int cash;

        System.out.println("[캐시백 계산]");

        System.out.print("결제 금액을 입력해주세요. (금액):");
        cash = sc.nextInt();
        int cashBack = ((cash/10)/100)*100;

        if (cashBack >= 300) {
            System.out.println("결제 금액은 " + cash + "원이고, 캐시백은 300원 입니다.");
        } else {
            System.out.println("결제 금액은 " + cash + "원이고, 캐시백은 " + cashBack + "원 입니다.");
        }

        }
    }

//date: 2024-04-12T16:42:38Z
//url: https://api.github.com/gists/8bb042d2d4168432074f73d3c5d7dfca
//owner: https://api.github.com/users/free8o0ter

import java.util.Scanner;
public class Main {
    public static void main(String[] args) {
        System.out.println("[캐시백 계산]");
        Scanner input = new Scanner(System.in);
        System.out.println("결제 금액을 입력해 주세요.(금액): ");
        int cash = input.nextInt();
        int cashBack = ((cash / 10)/100);
        if(cashBack > 3) cashBack = 3;

        System.out.println(String.format("결제 금액은 %d원이고, 캐시백은 %d원 입니다.",cash, cashBack*100));
    }
}
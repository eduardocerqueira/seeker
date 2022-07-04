//date: 2022-07-04T03:09:56Z
//url: https://api.github.com/gists/be4c51fefe1988d3ba13634fe7f88f28
//owner: https://api.github.com/users/yj0903


// 채유진
// 미니프로젝트 Quiz 2

import java.util.Scanner;

public class Quiz2 {
    public static void main(String[] args) {
        Scanner input = new Scanner(System.in);
        System.out.println("[캐시백 계산]");
        System.out.print("결제 금액을 입력해주세요. (금액) : ");
        int pay = Integer.parseInt(input.nextLine());

        int cashback = 0;
        if (pay >= 3000) {
            cashback = 300;
        } else {
            cashback = (pay / 1000)*100;
        }
        String s = String.format("결제금액은 %d원이고, 캐시백은 %d원입니다.",pay,cashback);
        System.out.println(s);
    }
}

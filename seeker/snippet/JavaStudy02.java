//date: 2023-05-02T16:47:44Z
//url: https://api.github.com/gists/9f6ee56b6b6c5d3427bb9d75879c7e58
//owner: https://api.github.com/users/leeyoungsans

/*
 이영산
 미니과제 2번 
 */
import java.util.Scanner;

public class JavaStudy02 {
    public static void main(String[] args){
        System.out.println("[캐시백 계산]");
        System.out.print("결제 금액을 입력해 주세요.(금액): ");
        Scanner sc = new Scanner(System.in);

        int input = sc.nextInt();
        int cashback = (int) (input * 0.1);
        if (cashback > 300){
            cashback = 300;
        }
        System.out.println(String.format("결재 금액은 %d원이고, 캐시백은 %d원입니다.",input,cashback));
    }
}
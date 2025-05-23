//date: 2025-05-23T16:47:13Z
//url: https://api.github.com/gists/9455a20c6d7fcdbeebd51e38aceb602b
//owner: https://api.github.com/users/rraon0714

import java.util.InputMismatchException;
import java.util.Scanner;

public class MyErrException {
    public static void main(String[] args) {
        
        int balance = 10000;
        Scanner sc = new Scanner(System.in);
        System.out.println("출금할 금액 입력: ");
        int withdraw = 0;


        try{
            withdraw = sc.nextInt();
            if (balance >= withdraw) {
                balance -= withdraw;
                System.out.println("출금 완료! 남은 잔액: " + balance);
            } else{
                System.out.println("잔액 부족!");
            }
        } catch(InputMismatchException e) {
            System.out.println("숫자를 입력하세요.");
        } catch(ArithmeticException e) {
            if (balance < withdraw)
             System.out.println("잔액 부족!");
        } finally{
            System.out.println("거래 기록이 저장되었습니다.");
        }
    }
}
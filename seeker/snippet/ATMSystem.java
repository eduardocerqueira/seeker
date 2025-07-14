//date: 2025-07-14T17:02:28Z
//url: https://api.github.com/gists/b5101685b2892e1c89c65f54e5a9aaff
//owner: https://api.github.com/users/moon-three

// 문서희
// ATM 시스템 (잔액 검사 및 출금 예외 처리)

import java.util.Scanner;

class InsufficientBalanceException extends RuntimeException {
    public InsufficientBalanceException(String message) {
        super(message);
    }
}

class Account {
    private String accountNumber;
    private int balance;

    public Account(String accountNumber, int balance) {
        this.accountNumber = accountNumber;
        this.balance = balance;
    }

    public void withdraw(int amount) {
        if(amount <= 0) {
            throw new IllegalArgumentException("출금액은 0보다 커야 합니다.");
        }
        if(balance < amount) {
            throw new InsufficientBalanceException("⚠️ 잔액 부족!");
        }
        balance -= amount;
    }

    public int getBalance() {
        return balance;
    }
}

public class ATMSystem {
    public static void main(String[] args) {

        Account account = new Account("12345", 10000);

        Scanner sc = new Scanner(System.in);

        System.out.print("출금할 금액 입력: ");
        String input = sc.nextLine();

        try {
            int price = Integer.parseInt(input);
            account.withdraw(price);
            System.out.println("✅ 출금 완료! 남은 잔액: " + account.getBalance());
        } catch (NumberFormatException e) {
            System.out.println("입력한 값이 숫자가 아닙니다.");
        } catch (IllegalArgumentException | InsufficientBalanceException e) {
            System.out.println(e.getMessage());
        } finally {
            System.out.println("💰 거래 기록이 저장되었습니다.");
            sc.close();
        }

    }
}
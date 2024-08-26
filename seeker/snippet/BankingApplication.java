//date: 2024-08-26T16:49:19Z
//url: https://api.github.com/gists/b29aeaabf6b03dcde670c3c7b18786b3
//owner: https://api.github.com/users/RamshaMohammed

import java.util.Scanner;

class LessBalanceException extends Exception {
    public LessBalanceException()
    {
        System.out.println("lessBalance exception");
    }
}

public class BankingApplication {

    public static void main(String[] args) {
        BankAccount account = new BankAccount(10000); 
        try {
            account.deposit(5000); 
            account.checkBalance();
            account.withdraw(3500); 
            account.checkBalance();
            account.withdraw(900); 

        } catch (LessBalanceException e) {
            System.out.println("Exception caught: " + e.getMessage());
        } finally {
            System.out.println("Final balance is:");
            account.checkBalance();
        }
    }
}

class BankAccount {
    private double balance;

    public BankAccount(double initialBalance) {
        this.balance = initialBalance;
    }

    public void deposit(double amount) {
        if (amount > 0) {
            balance += amount;
            System.out.println("Successfully deposited " + amount);
        } else {
            System.out.println("Deposit amount is");
        }
    }

    public void withdraw(double amount) throws LessBalanceException {
        if (amount > balance) {
            throw new LessBalanceException();
        } else if (amount <= 0) {
            System.out.println("Withdrawal amount is");
        } else {
            balance -= amount;
            System.out.println("Successfully withdrwn: " + amount);
        }
    }

    public void checkBalance() {
        System.out.println("Current balance: " + balance);
    }
}
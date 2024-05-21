//date: 2024-05-21T16:59:58Z
//url: https://api.github.com/gists/3ef525c8c2f30b79890baaadf6102112
//owner: https://api.github.com/users/maulanafanny

// Refactored Version

interface Account {
    void deposit(double amount);
    void withdraw(double amount);
    double getBalance();
}

abstract class AbstractAccount implements Account {
    protected double balance;

    public AbstractAccount(double balance) {
        this.balance = balance;
    }

    public void deposit(double amount) {
        balance += amount;
    }

    public void withdraw(double amount) {
        if (amount > balance) {
            System.out.println("Insufficient funds");
        } else {
            balance -= amount;
        }
    }

    public double getBalance() {
        return balance;
    }
}

class SavingsAccount extends AbstractAccount {
    public SavingsAccount(double balance) {
        super(balance);
    }

    public void addInterest(double rate) {
        balance *= (1 + rate);
    }
}

class CheckingAccount extends AbstractAccount {
    public CheckingAccount(double balance) {
        super(balance);
    }

    @Override
    public void withdraw(double amount) {
        if (amount > balance) {
            System.out.println("Insufficient funds");
        } else {
            balance -= amount;
            System.out.println("Withdrawal successful");
        }
    }
}

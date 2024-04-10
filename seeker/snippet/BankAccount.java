//date: 2024-04-10T16:43:49Z
//url: https://api.github.com/gists/1d9d10185025e7a33d3f397add38c18a
//owner: https://api.github.com/users/Amin-Elgazar

package org.example.banks;

import java.util.Date;

public class BankAccount {
    protected static int numAccounts;
    protected String name;
    protected double balance;
    protected int acctNum;
    protected Date date;

    public BankAccount(String name) {
        this.name = name;
        acctNum = generateAcctNum();
        date = new Date();
        numAccounts++;
    }

    public static int getNumAccounts() {
        return numAccounts;
    }

    public double getBalance() {
        return balance;
    }

    public int getAccountNumber() {
        return acctNum;
    }

    public boolean deposit(double amt) {
        if (amt < 0) return false;
        this.balance += amt;
        return true;
    }

    public boolean withdraw(double amt) {
        if (amt < 0 || amt > this.balance) return false;
        this.balance -= amt;
        return true;
    }

    public boolean transfer(BankAccount that, double amt) {
        if (that == null || amt < 0 || amt > this.balance) return false;
        this.balance -= amt;
        that.balance += amt;
        return true;
    }

    public String toString() {
        return name + " [" + acctNum + "]\n" + date + "\n" + String.format("$%,.2f", balance);
    }

    public boolean equals(BankAccount that) {
        return this.acctNum == that.acctNum;
    }

    public int compareTo(BankAccount that) {
        return this.acctNum - that.acctNum;
    }

    private int generateAcctNum() {
        return (int) (Math.random() * 900000000) + 100000000;
    }


}
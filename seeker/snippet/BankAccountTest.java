//date: 2024-04-10T16:43:49Z
//url: https://api.github.com/gists/1d9d10185025e7a33d3f397add38c18a
//owner: https://api.github.com/users/Amin-Elgazar

package org.example.banks;

import org.junit.Before;
import org.junit.Test;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;

import static org.junit.Assert.*;

public class BankAccountTest {
    private BankAccount account1;
    private BankAccount account2;
    private BankAccount a1, a2, a3;
    private final PrintStream standardOut = System.out;
    private final ByteArrayOutputStream outputStreamCaptor = new ByteArrayOutputStream();


    @Before
    public void setUp() {
        a1 = new BankAccount("Mike V");
        a2 = new BankAccount("Fred M");
        a3 = a1;
        System.setOut(new PrintStream(outputStreamCaptor));
    }

    @Test
    public void testBankAccount() {
        assertTrue("expect name to be set properly", a1.toString().contains("Mike V"));
        assertEquals("expect starting balance to be 0", a1.getBalance(), 0, 0.01);
        assertTrue("expect name to be set properly", a2.toString().contains("Fred M"));
        assertEquals("expect starting balance to be 0", a2.getBalance(), 0, 0.01);
    }

    @Test
    public void testDepositValid() {
        assertTrue("expect valid deposit to return true", a1.deposit(5.55));
        assertEquals("expect valid deposit to update balance", a1.getBalance(), 5.55, 0.01);
    }

    @Test
    public void testWithdrawValid() {
        a1.deposit(5.55);
        assertTrue("expect valid withdraw to return true", a1.withdraw(5));
        assertEquals("expect valid withdraw to update balance", a1.getBalance(), 0.55, 0.01);
    }

    @Test
    public void testWithdrawTooMuch() {
        a1.deposit(10);
        assertFalse("expect too high of withdraw to return false", a1.withdraw(12));
        assertEquals("expect invalid withdraw not to change balance", a1.getBalance(), 10, 0.01);
    }


    @Test
    public void testToString() {
        assertTrue("expect toString to contain the name", a1.toString().contains("Mike V"));
        assertTrue("expect toString to contain dollar sign", a1.toString().contains("$"));
        assertTrue("expect toString to contain square brackets", a1.toString().contains("["));
        assertTrue("expect toString to contain the time stamp", a1.toString().contains(":"));
    }

    @Test
    public void testToStringRounding() {
        a1.deposit(99.988);
        String s = a1.toString();
        assertTrue("expect balance to be rounded to two places in toString", s.contains("99.99"));
    }

    @Test
    public void testSystemOut() {
        a1.deposit(100);
        a1.withdraw(10);
        a1.deposit(-10);
        a1.withdraw(-10);
        a1.withdraw(100000);
        a1.toString();
        String out = outputStreamCaptor.toString().trim();
        assertTrue("methods should not print anything", out == null || out.equals(""));
    }

    @Test
    public void testGenerateAcctNum() {
        String s = a1.toString();
        String n = s.substring(s.indexOf("[") + 1, s.indexOf("]"));
        assertEquals("expect account number to contain 9 digits.", 9, n.length());
    }

    @Test
    public void testGetNumAccounts() {
        int x = BankAccount.getNumAccounts();
        new BankAccount("");
        new BankAccount("");
        assertEquals("expect numAccounts to track total number of accounts", x + 2, BankAccount.getNumAccounts());
    }

    @Test
    public void testEquals1() {
        assertEquals("expect equals to return true for same object", a1, a1);
    }

    @Test
    public void testCompareTo1() {
        if (a1.getAccountNumber() > a2.getAccountNumber()) {
            BankAccount temp = a1;
            a1 = a2;
            a2 = temp;
        }
        assertTrue("expect negative for in-order comparison", a1.compareTo(a2) < 0);
    }

    @Test
    public void testCompareTo2() {
        if (a1.getAccountNumber() > a2.getAccountNumber()) {
            BankAccount temp = a1;
            a1 = a2;
            a2 = temp;
        }
        assertTrue("expect positive for out-of-order comparison", a2.compareTo(a1) > 0);
    }

    @Test
    public void testTransferValidRequest1() {
        a1.deposit(10);
        a2.deposit((5));
        assertTrue("expect valid transfer to return true", a1.transfer(a2, 7));
    }

    @Test
    public void testTransferValidRequest3() {
        a1.deposit(10);
        a2.deposit((5));
        a1.transfer(a2, 7);
        assertEquals("expect valid transfer to increase destination balance", a2.getBalance(), 12, 0.01);
    }

    @Test
    public void testTransferInvalidRequest1() {
        a1.deposit(10);
        a2.deposit((5));
        assertFalse("expect invalid transfer to return false", a2.transfer(a1, 7));
    }

    @Before
    public void setitUp() {
        account1 = new BankAccount("Alice");
        account2 = new BankAccount("Bob");
    }

    @Test
    public void testGettNumAccounts() {
        int initialNumAccounts = BankAccount.getNumAccounts();
        new BankAccount("Charlie");
        assertEquals(initialNumAccounts + 1, BankAccount.getNumAccounts());
    }

    @Test
    public void testGetBalance() {
        account1.deposit(100);
        assertEquals(100, account1.getBalance(), 0.01);
    }

    @Test
    public void testDeposit() {
        assertTrue(account1.deposit(100));
        assertEquals(100, account1.getBalance(), 0.01);

        assertFalse(account1.deposit(-50));
    }

    @Test
    public void testWithdraw() {
        account1.deposit(100);
        assertTrue(account1.withdraw(50));
        assertEquals(50, account1.getBalance(), 0.01);

        assertFalse(account1.withdraw(-50));
        assertFalse(account1.withdraw(100));
    }

    @Test
    public void testTransfer() {
        account1.deposit(100);
        assertTrue(account1.transfer(account2, 50));
        assertEquals(50, account1.getBalance(), 0.01);
        assertEquals(50, account2.getBalance(), 0.01);

        assertFalse(account1.transfer(account2, -50));
        assertFalse(account1.transfer(account2, 100));
        assertFalse(account1.transfer(null, 50));
    }

}


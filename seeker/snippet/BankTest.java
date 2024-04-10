//date: 2024-04-10T16:43:49Z
//url: https://api.github.com/gists/1d9d10185025e7a33d3f397add38c18a
//owner: https://api.github.com/users/Amin-Elgazar

package org.example.banks;

import org.junit.Before;
import org.junit.Test;
import static org.junit.Assert.*;

public class BankTest  {

    private BankAccount a1, a2, a3, a4;
    private Bank b;

    @Before
    public void setUp() {
        a1 = new BankAccount("Mike V");
        a2 = new BankAccount("Fred M");
        a3 = new BankAccount("Cleetus M");
        a4 = new BankAccount("Earl B");
        b = new Bank(3);
    }
    @Test
    public void testFind1() {
        int acct = a1.getAccountNumber();
        assertNull("expect null return for finding nonexistent account", b.find(acct));
        b.add(a1);
        assertEquals("expect reference returned for finding existing account", b.find(acct), a1);
    }
    @Test
    public void testContains2() {
        b.add(a1);
        assertTrue("expect true for just-added account", b.contains(a1));
    }
    @Test
    public void testContains3() {
        b.add(a1);
        b.remove(a1);
        assertFalse("expect false for just-removed account", b.contains(a1));
    }
    @Test
    public void testBank() {
        assertTrue("expect new Bank to have zero accounts", b.getCount() == 0);
    }
    @Test
    public void testAddDuplicate() {
        b.add(a1);
        b.add(a1);
        assertTrue("expect adding duplicate account not to change the count", b.getCount() == 1);
    }
    @Test
    public void testAddOverflow1() {
        assertTrue("expect adding a new account to return true", b.add(a1));
        assertTrue("expect adding a new account to return true", b.add(a2));
        assertTrue("expect adding a new account to return true", b.add(a3));
        assertTrue("expect adding account over initial capacity to work", b.add(a4));
    }
    @Test
    public void testRemoveNormalSingleItem1() {
        b.add(a1);
        assertTrue("expect removal of existing account to return true", b.remove(a1));
    }
    @Test
    public void testRemoveNormalSingleItem2() {
        b.add(a1);
        assertTrue("expect removal of existing account to return true", b.remove(a1));
        assertTrue("expect removal to decrement count properly", b.getCount() == 0);
    }
    @Test
    public void testRemoveNonExistentSingleItem() {
        assertFalse("expect removal of nonexistent account from empty bank to return false", b.remove(a1));
    }
    @Test
    public void testToString() {
        b.add(a1);
        assertNotNull("expect output for toString", b.toString());
    }
    @Test
    public void testSortAccounts() {
        Bank bank = new Bank(10);
        BankAccount account1 = new BankAccount("Amin");
        BankAccount account2 = new BankAccount("Mike");

        account1.acctNum = 1002;
        account2.acctNum = 1001;

        bank.add(account1);
        bank.add(account2);

        bank.sort();

        assertEquals("Account 2 should be first after sorting", account2, bank.find(account2.getAccountNumber()));
        assertEquals("Account 1 should be second after sorting", account1, bank.find(account1.getAccountNumber()));


    }
}
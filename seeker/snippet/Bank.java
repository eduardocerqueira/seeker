//date: 2024-04-10T16:43:49Z
//url: https://api.github.com/gists/1d9d10185025e7a33d3f397add38c18a
//owner: https://api.github.com/users/Amin-Elgazar

package org.example.banks;

//
public class Bank {
    private BankAccount[] accounts;
    private int count;

    public Bank(int cap) {
        accounts = new BankAccount[cap];
        count = 0;
    }

    public boolean add(BankAccount a) {
        if (contains(a)) return false;
        if (full()) resize(2 * count);
        accounts[count++] = a;
        return true;
    }

    public boolean remove(BankAccount a) {
        if (!contains(a)) return false;
        accounts[indexOf(a)] = accounts[--count];
        if (count <= accounts.length / 4) resize(count / 2);
        return true;
    }

    public boolean contains(BankAccount a) {
        return indexOf(a) != -1;
    }


    public BankAccount find(int acct) {
        for (int i = 0; i < count; i++)
            if (accounts[i].getAccountNumber() == acct) return accounts[i];
        return null;
    }

    private int indexOf(BankAccount a) {
        if (a == null) return -1;
        for (int i = 0; i < count; i++)
            if (accounts[i].equals(a)) return i;
        return -1;
    }

    public int getCount() {
        return count;
    }

    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Bank Accounts\n");
        for (int i = 0; i < count; i++) {
            sb.append(accounts[i].toString());
            sb.append("\n    **************    \n");
        }
        return sb.toString();
    }

    public void sort() {
        for (int i = 1; i < count; i++) {
            BankAccount temp = accounts[i];
            int j;
            for (j = i - 1; j >= 0 && less(temp, accounts[j]); j--) {
                accounts[j + 1] = accounts[j];
            }
            accounts[j + 1] = temp;
        }
    }

    private void resize(int newSize) {
        BankAccount[] a = new BankAccount[newSize];
        System.arraycopy(accounts, 0, a, 0, count);
        accounts = a;
    }

    private boolean less(BankAccount temp, BankAccount j) {
        return temp.getAccountNumber() < j.getAccountNumber();
    }

    private boolean full() {
        return count == accounts.length;
    }
}
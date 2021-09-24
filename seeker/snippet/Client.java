//date: 2021-09-24T16:50:05Z
//url: https://api.github.com/gists/ed0b8cef51379aeb9036644863cd63e1
//owner: https://api.github.com/users/Menelaii

public class NaturalClient implements IClient
{
    private String name;
    private String surname;
    private final int passportSeries;
    private final int passportNumber;
    private final ArrayList<Account> accounts;

    public NaturalClient(String name, String surname, int passportSeries, int passportNumber)
    {
        this.name = name;
        this.surname = surname;
        this.passportSeries = passportSeries;
        this.passportNumber = passportNumber;

        accounts = new ArrayList<Account>();
    }

    public NaturalClient(String name, String surname, int passportSeries, int passportNumber, ArrayList<Account> accounts)
    {
        this.name = name;
        this.surname = surname;
        this.passportSeries = passportSeries;
        this.passportNumber = passportNumber;

        this.accounts = accounts;
    }

    public Account getAccountWithIdentifier(int identifier)
    {
        for (Account account : accounts)
        {
            if (identifier == account.getIdentifier())
            {
                return account;
            }
        }

        throw new IllegalArgumentException();
    }

    public ArrayList<Account> getAccounts()
    {
        return  accounts;
    }

    @Override
    public ArrayList<Account> getDebitAccounts()
    {
        ArrayList<Account> debitAccounts = new ArrayList<Account>();
        for (Account account : accounts)
        {
            if(account instanceof DebitAccount)
            {
                debitAccounts.add(account);
            }
        }

        return debitAccounts;
    }

    @Override
    public ArrayList<Account> getCreditAccounts()
    {
        ArrayList<Account> creditAccounts = new ArrayList<Account>();
        for (Account account : accounts)
        {
            if(account instanceof CreditAccount)
            {
                creditAccounts.add(account);
            }
        }

        return creditAccounts;
    }

    @Override
    public double getBalanceFromAllDebitAccounts()
    {
        double sum = 0;
        for (Account account : accounts)
        {
            if(account instanceof DebitAccount)
            {
                sum += account.getBalance();
            }
        }

        return sum;
    }

    @Override
    public double getDebts()
    {
        double debt = 0;
        for (Account account : accounts)
        {
            double balance = account.getBalance();
            debt += balance < 0 ? balance : 0;

            if(account instanceof CreditAccount)
            {
                CreditAccount creditAccount = (CreditAccount) account;
                debt += creditAccount.getAccruedCommission();
                debt += creditAccount.getInterestCharges();
            }
        }

        return debt;
    }

    public double getBalanceFromAllAccounts()
    {
        double sum = 0;
        for (Account account : accounts)
        {
            sum += account.getBalance();
        }

        return sum;
    }

    public ArrayList<Account> getAccountsWithPositiveBalance()
    {
        ArrayList<Account> accountsWithPositiveBalance = new ArrayList<Account>();
        for (Account account : accounts)
        {
            if(account.getBalance() >= 0)
            {
                accountsWithPositiveBalance.add(account);
            }
        }

        return accountsWithPositiveBalance;
    }

    public void deleteAccountWithIdentifier(int identifier)
    {
        for (Account account : accounts)
        {
            if(account.getIdentifier() == identifier)
            {
                accounts.remove(account);
                return;
            }
        }
    }

    public void addAccount(Account account)
    {
        accounts.add(account);
    }

    public void takeMoneyFromAccount(Account account, double amount) throws InsufficientFundsException
    {
        if(account.getBalance() < amount)
            throw new InsufficientFundsException("Недостаточно средств");

        account.takeFromBalance(amount);
    }

    @Override
    public void addMoneyToAccount(Account account, double amount)
    {
        account.addToBalance(amount);
    }
}
//date: 2021-10-18T17:06:25Z
//url: https://api.github.com/gists/ad0ad6e6e5543415debb135712346621
//owner: https://api.github.com/users/anelaco

public class BankAccount {
    private String customerId;  // letter and three numbers
    private int accountNumber;  // 5 digits
    private double initialBalance; // above 1000

    public BankAccount(){
        customerId = "None";
        accountNumber = 0;
        initialBalance = 0;
    }

    /**
     * @param _customerId of class BankAccount
     * @param _accountNumber of class BankAccount
     * @param _initialBalance of class BankAccount
     * @throws CustomerIdException if the setCustomerId method fails
     * @throws LenghtOfAccountNumberException if the setAccountNumber method fails
     * @throws InitialBalanceException if the setInitialBalance method fails
     */
    public BankAccount(String _customerId, int _accountNumber, double _initialBalance)
            throws CustomerIdException, LenghtOfAccountNumberException, InitialBalanceException {
        setCustomerId(_customerId);
        setAccountNumber(_accountNumber);
        setInitialBalance(_initialBalance);
    }

    public BankAccount(BankAccount originalBankAccount){
        if(originalBankAccount == null){
            System.out.println("Not valid");
            System.exit(0);
        }else{
            customerId = originalBankAccount.customerId;
            accountNumber = originalBankAccount.accountNumber;
            initialBalance = originalBankAccount.initialBalance;
        }
    }

    /**
     * @param _customerId of class BankAccount
     * @throws CustomerIdException if the inputted _customerId is not valid when checked by the method isValidId()
     */
    public void setCustomerId(String _customerId) throws CustomerIdException {
            if (isValidId(_customerId))
                this.customerId = _customerId;
            else
                throw new CustomerIdException("Your ID is invalid!");
    }

    /**
     * @param _accountNumber of class BankAccount
     * @throws LenghtOfAccountNumberException if the inputted _accountNumber is not valid when checked
     * by the method isValidNumber()
     */
    public void setAccountNumber(int _accountNumber) throws LenghtOfAccountNumberException {
        if (isValidNumber(_accountNumber))
            this.accountNumber = _accountNumber;
        else
            throw new LenghtOfAccountNumberException("This is not the right amount of numbers!");
    }

    /**
     * @param _initialBalance of class BankAccount
     * @throws InitialBalanceException if the inputted _initialBalance is not valid when checked
     * by the method isValidBalance()
     */
    public void setInitialBalance(double _initialBalance) throws InitialBalanceException {
            if (isValidBalance(_initialBalance))
                this.initialBalance = _initialBalance;
            else
                throw new InitialBalanceException("Balance must be over 1000$!");
    }

    public String getCustomerId() {
        return customerId;
    }

    public int getAccountNumber() {
        return accountNumber;
    }

    public double getInitialBalance() {
        return initialBalance;
    }

    public String toString(){
        return ("Customer ID is: " + customerId + "  Account number: " + accountNumber + "  Balance is: " + initialBalance);
    }

    public boolean equals(Object otherObject){
        if(otherObject == null)
            return false;
        else if(getClass() != otherObject.getClass())
            return false;
        else{
            BankAccount otherBankAccount = (BankAccount)otherObject;
            return(this.customerId.equalsIgnoreCase(otherBankAccount.customerId)&&
                    this.accountNumber == otherBankAccount.accountNumber &&
                    this.initialBalance == otherBankAccount.initialBalance);
        }

    }

    /**
     * Checks if the given requirement(customer ID  must contain a letter at the beginning and be followed by three numbers)
     * is met for the customer ID
     * @param _customerId of class BankAccount
     * @return true if the requirement for customer ID is met, otherwise false
     */
    public boolean isValidId(String _customerId){
        int lengthOfId = 4;
        int numberOfDigits = 3;
        int isTrue = 0;
        if(_customerId.length() == lengthOfId)
            if(Character.isLetter(_customerId.charAt(0)))
                for(int i = 1; i < lengthOfId; i++)
                    if(Character.isDigit(_customerId.charAt(i)))
                        isTrue++;

        return isTrue == numberOfDigits;
    }

    /**
     * Checks if the given requirement(accountNumber must contain only 5 numbers)
     * is met for the account number
     * @param _accountNumber of class BankAccount
     * @return true if the requirement for account number is met, otherwise false
     */
    public boolean isValidNumber(int _accountNumber){
        int lenghtOfAccountNumber = String.valueOf(_accountNumber).length();
        return lenghtOfAccountNumber == 5;
    }

    /**
     * Checks if the given requirement(initialBalance must be over 1000)
     * is met for the initial balance
     * @param _initialBalance of class BankAccount
     * @return true if the requirement for initial balance is met, otherwise false
     */
    public boolean isValidBalance(double _initialBalance){
        return _initialBalance > 1000;
    }
}
//date: 2023-12-01T16:58:06Z
//url: https://api.github.com/gists/be28ceb8358334ed15f3153d2e160301
//owner: https://api.github.com/users/deanMoisakos

package lab5;
/**
*
* @author DEAN
*/
public class Lab5Part3 {
    public static void main(String[] args) {
    Account account = new Account();
    System.out.println("The balance of the account is: "+account.getBalance());
    System.out.println("The monthly interest is:
    "+String.format("%.2f",account.getMonthlyInterest()));
    System.out.println("This account was created on: "+account.getDate());
  }
}
                       
class Account{
  private int id;
  private double balance;
  private double annualInterestRate;
  private java.util.Date dateCreated;
  public Account(){
  this(123456,30000.00, 5.5);
}
  
public Account(int id, double balance, double interest ){
  this.id = id;
  this.balance = balance;
  this.annualInterestRate= interest;
  dateCreated = new java.util.Date();
}
  
public double getMonthlyInterestRate(){
  return (annualInterestRate/100)/12;
}
  
public double getMonthlyInterest(){
  return balance*((annualInterestRate/100)/12);
}
  
public double withdraw(){
  this.balance = balance-1500;
  return balance;
}
  
public double deposit(){
  this.balance = balance + 5000;
  return balance;
}
  
public double getBalance(){
  deposit();
  withdraw();
  return this.balance;
}
  
public java.util.Date getDate(){
  return dateCreated;
}
  
public void setNew(int id, double balance, double interest){
  this.id = id;
  this.balance = balance;
  this.annualInterestRate= interest;
}
  
}
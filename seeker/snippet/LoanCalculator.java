//date: 2023-04-26T17:09:45Z
//url: https://api.github.com/gists/11fd5e52170ae96380eb1589ae3d11ae
//owner: https://api.github.com/users/hmc-hassan

package Operators;
import java.util.Scanner;

/*
Case:
To Qualify for a loan, a person must make at least $30,000 and have been working at
their current job for at least 2 years.
 */
public class LoanCalculator {
    public static void main (String args[])
    {
        //1. What we Know
        int salary = 30000;
        double tenure = 2;

        //2. What we don't know
        //2.1 Get the employee salary
        System.out.println("Enter the gross salary of the employee!");
        Scanner sc = new Scanner(System.in);
        double salaryearned = sc.nextDouble();

        //2.2 Get the Years of job
        System.out.println("Enter the year of employment at current company for the employee!");
        double tyear = sc.nextDouble();
        sc.close();


        //3. Decision making

        if (salaryearned >= salary && tyear >= tenure) {
                System.out.println("Congratulations! You are eligible for the loan.");
            }
        else{
            System.out.println("Sorry! You are ineligible for the loan.");
        }
    }
}

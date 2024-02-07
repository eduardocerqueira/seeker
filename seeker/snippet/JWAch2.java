//date: 2024-02-07T16:54:33Z
//url: https://api.github.com/gists/3029760ac88460fbe37766a80b5b709a
//owner: https://api.github.com/users/jwabt2001

/*Jared Abt
 *jwabt2001@student.stcc.edu
 *2/7/2024
 *CSC-111-D01
 *Chapter 2 Project
 *Calculating initial deposit using final amount, annual interest rate and time
*/

package ch2;

import java.util.Scanner;

public class ch2 {

	public static void main(String[] args) {
		
		Scanner input = new Scanner(System.in);
		
		System.out.println("Enter final account value: ");
		double value = input.nextDouble();
		
		System.out.println("Enter annual interest rate in percent: ");
		double yearlyInterest = input.nextDouble();
		
		System.out.println("Enter number of years: ");
		double years = input.nextDouble();
		
		double months = years * 12;
		double monthlyInterest = yearlyInterest / 12;
		
		double deposit = value / Math.pow((1 + monthlyInterest), months);
		System.out.println("Initial deposit value is: ");
		System.out.print(deposit);
		input.close();
	}

}
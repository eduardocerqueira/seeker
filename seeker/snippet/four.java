//date: 2023-09-28T16:58:12Z
//url: https://api.github.com/gists/9cf6deb52bc6c31ec859a34b3b6146e1
//owner: https://api.github.com/users/AyushKumarShukla

/**
This program converts temperature in Fahrenheit to celcius using the formula: 
C = (F-32)/1.8
*/

import java.util.Scanner;

public class four
{
	public static void main(String args[])
	{
		System.out.printf("****FAHRENHEIT TO CELCIUS CONVERTER****\n");
		Scanner scan=new Scanner(System.in);
		System.out.printf("Enter temperature in Fahrenheit: ");
		float temp_in_fahr=scan.nextFloat();
		float temp_in_cel=(float)((temp_in_fahr-32)/1.8);
		System.out.printf("Temperature in Celcius:%.2f \n",temp_in_cel);

	}
}

/*
OUTPUT
SET 1:
****FAHRENHEIT TO CELCIUS CONVERTER****
Enter temperature in Fahrenheit: -40
Temperature in Celcius:-40.00
SET 2:
****FAHRENHEIT TO CELCIUS CONVERTER****
Enter temperature in Fahrenheit: 32
Temperature in Celcius:0.00
*/
/** A program to determine sum of the following series for given value of n:(1 + 1/2 + 1/3 + 
…… + 1/n)  up to two decimal places
*/
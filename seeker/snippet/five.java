//date: 2023-09-28T16:58:12Z
//url: https://api.github.com/gists/9cf6deb52bc6c31ec859a34b3b6146e1
//owner: https://api.github.com/users/AyushKumarShukla

import java.util.Scanner;

public class five
{
	public static void main(String args[])
	{
		Scanner scan=new Scanner(System.in);
		System.out.printf("Enter a value of n: ");
		long terms=scan.nextLong();
		float sum=0f;
		for(int i=1;i<=terms;i++)
		{
			sum+=(1.0f/i);
		}
		System.out.printf("Computed Value Of Series: %f\n",sum);

	}
}
/*Limit of the series found to be 15.403683 approximately*/
/*
OUTPUT
SET 1:
Enter a value of n: 5
Computed Value Of Series: 2.283334
SET 2
Enter a value of n: 2
Computed Value Of Series: 1.500000
*/
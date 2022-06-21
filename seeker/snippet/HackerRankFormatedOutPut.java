//date: 2022-06-21T17:00:07Z
//url: https://api.github.com/gists/0bcee9ed3376072fa71d206e3393a7c1
//owner: https://api.github.com/users/carolineferraz

import java.util.Scanner;

public class HackerRankFormatedOutput {
	
public static void main(String[] args) {
		
        Scanner sc=new Scanner(System.in);
        
        System.out.println("================================");
        
        for(int i=0;i<3;i++)
        {
            String s1=sc.next();
            int x=sc.nextInt();
            System.out.printf("%-15s%03d%n",s1,x);
        }
        
        System.out.println("================================");

}

}

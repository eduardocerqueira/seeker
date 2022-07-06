//date: 2022-07-06T16:46:26Z
//url: https://api.github.com/gists/d5dc2e4ddd0fdef24775ae83260eb408
//owner: https://api.github.com/users/mahisrivastava1218

import java.util.*;
import java.lang.*;
import java.io.*;

/* Name of the class has to be "Main" only if the class is public. */
class Codechef
{
	public static void main (String[] args) throws java.lang.Exception
	{
	 Scanner scn = new Scanner(System.in);
	 int t=scn.nextInt();
     
     for(int text=0;text<t;text++){
         int n=scn.nextInt();
         int m=scn.nextInt();
         int res=0;
         for(int i=0;i<2;i++){
         res=n*m;
         }
	    System.out.println(res);
     }
    
   }
}

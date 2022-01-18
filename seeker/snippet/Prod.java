//date: 2022-01-18T17:14:46Z
//url: https://api.github.com/gists/acd1c334ea4a867e3b9817086b3d2870
//owner: https://api.github.com/users/Arjun2002tiwari

import java.util.Scanner;

public class Prod{
    public static void main(String[] args) {
        Scanner sc=new Scanner(System.in);
		int n=sc.nextInt();

		int max=0;
		int num=1;

		for(int i=0;i<n;i++){
			int a=sc.nextInt();
			int b=sc.nextInt();

			if((a-b)>0){
				int dif=a-b;
				if(dif>max){
				max=Math.max(dif,max);
				num=1;
				}
			}
			else if((b-a>0)){
				int dif=b-a;
				if(dif>max){
				max=Math.max(dif, max);
				num=2;
				}
			}

		}
		System.out.print(num+" ");
		System.out.print(max);
		
    }
}
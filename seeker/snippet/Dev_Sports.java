//date: 2022-04-18T17:05:19Z
//url: https://api.github.com/gists/bc47901e998ac63013b025d3945f5892
//owner: https://api.github.com/users/pratik8912

import java.util.Scanner;

public class Dev_Sports {
    public static void main(String[] args) throws java.lang.Exception {

        Scanner sc= new Scanner(System.in);

        int t=sc.nextInt();

        for(int i=0;i<t;i++){
            int z=sc.nextInt();
            int y=sc.nextInt();
            int a=sc.nextInt();
            int b=sc.nextInt();
            int c=sc.nextInt();

            if(z-y>=a+b+c)
                System.out.println("Yes");
            else
                System.out.println("No");
        }

    }
}

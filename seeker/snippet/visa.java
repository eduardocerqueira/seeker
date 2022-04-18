//date: 2022-04-18T17:05:32Z
//url: https://api.github.com/gists/3e476966d4478fac3d41a05239c71a9c
//owner: https://api.github.com/users/pratik8912

import java.util.Scanner;

public class visa {
    public static void main(String[] args) throws java.lang.Exception {

        Scanner sc=new Scanner(System.in);

        int t=sc.nextInt();

        while(t>0){
            int x1=sc.nextInt();
            int x2=sc.nextInt();
            int y1=sc.nextInt();
            int y2=sc.nextInt();
            int z1=sc.nextInt();
            int z2=sc.nextInt();

            if(x1<=x2 && y1<=y2 && z1>=z2)
                System.out.println("Yes");
            else
                System.out.println("No");


            t--;
        }

    }
}

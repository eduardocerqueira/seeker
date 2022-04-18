//date: 2022-04-18T17:04:48Z
//url: https://api.github.com/gists/34dc4b7cf7c2f229c4ce5f36d997ff6f
//owner: https://api.github.com/users/pratik8912

import java.util.Scanner;

public class Problem_Category {
    public static void main(String[] args) throws java.lang.Exception {

        Scanner sc =new Scanner(System.in);
        int T=sc.nextInt();
        while(T-->0){
            int x= sc.nextInt();
            if(x>=1 && x<100)
                System.out.println("Easy");
            else if(x>=100 && x<200)
                System.out.println("Medium");
            else
                System.out.println("Hard");
        }

    }
}

//date: 2024-04-22T16:55:58Z
//url: https://api.github.com/gists/b70fc6fae7c2272d22dcfc7f84b23619
//owner: https://api.github.com/users/Aishwarya2233

import java.util.Scanner;
public class LargestNum {
    public static void main(String[] args){
        System.out.println("Enter any 2 integers: ");
        Scanner sc= new Scanner(System.in);
        int num1 = sc.nextInt();
        int num2 = sc.nextInt();
        if(num1 > num2){
            System.out.println(num1);
        } else{
            System.out.println(num2);
        }
    }
}

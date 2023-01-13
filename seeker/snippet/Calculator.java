//date: 2023-01-13T17:08:59Z
//url: https://api.github.com/gists/3d46061af085121a0b47883e63e0360e
//owner: https://api.github.com/users/akbarsiddique

import java.util.*;
public class Calculator {
public static void main(String[] args){
    Scanner sc = new Scanner(System.in);
    System.out.println("Enter  a :");
    int a= sc.nextInt();
    System.out.println("Enter b :");
    int b=  sc.nextInt();
    System.out.println("Enter opretor :");
    char operator = sc.next().charAt(0);
    switch (operator){
        case '+':
            System.out.println(a+b);
            break;
        case '-':
            System.out.println(a-b);
            break;
        case '*':
            System.out.println(a*b);
            break;
        case '/':
            System.out.println(a/b);
            break;
        case '%':
            System.out.println(a%b);
            break;

    }

}
}
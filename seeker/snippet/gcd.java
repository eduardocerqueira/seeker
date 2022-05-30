//date: 2022-05-30T16:58:43Z
//url: https://api.github.com/gists/0d6b0f5a9078ecff708e9919366c7c81
//owner: https://api.github.com/users/sangeet

import java.util.*;

class Program {
    public static void main(String[] args) {
        Gcd test1 = new Gcd();
        test1.accept();
        test1.display();
    }
}

class Gcd {
    int num1;
    int num2;

    Gcd() {
        num1 = 0;
        num2 = 0;
    }

    void accept() {
        Scanner sc = new Scanner(System.in);

        System.out.println("Enter number 1:");
        num1 = sc.nextInt();

        System.out.println("Enter number 2:");
        num2 = sc.nextInt();
    }

    int gcd(int x, int y) {
        if (y == 0) {
            return x;
        } else {
            return gcd (y, x % y);
        }
    }

    void display() {
        System.out.println("The GCD of " + num1 + " and " + num2 + " is " + gcd(num1, num2));
    }
}

//date: 2024-10-03T17:01:53Z
//url: https://api.github.com/gists/f151d6966f8e5557ad6dc5a4643d5721
//owner: https://api.github.com/users/mukhushi24

package Oct.ex_0302024Calculator;
public class Lab_002 {
    public static void main(String[] args) {
        System.out.println("Welcome to the Simple Calculator!");

        int num1 = 10;
        int num2 = 5;
        char operator = '+';

        int result = 0;

        switch (operator) {
            case '+':
                result = num1 + num2;
                break;
            case '-':
                result = num1 - num2;
                break;
            case '*':
                result = num1 * num2;
                break;
            case '/':
                if (num2 != 0) {
                    result = num1 / num2;
                } else {
                    System.out.println("Error: Cannot divide by zero!");
                    return;
                }
                break;
            default:
                System.out.println("Invalid operator");
                return;
        }

        System.out.println("Result:");
        System.out.println(num1 + " " + operator + " " + num2 + " = " + result);
    }
}



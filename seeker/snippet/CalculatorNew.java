//date: 2021-12-23T17:08:36Z
//url: https://api.github.com/gists/36c30388f08d73292601ee250ec67991
//owner: https://api.github.com/users/rrajesh1979

import java.util.Stack;

public class CalculatorNew {
    /* function to parse input string like 1 plus 5 minus 4 and return the result */
    public static int calculate(String input) {
        int result = 0;

        // create a stack to store the operands
        Stack<Integer> operands = new Stack<Integer>();

        // create a stack to store the operators
        Stack<String> operators = new Stack<String>();

        // split the input string into tokens
        String[] tokens = input.split(" ");

        // iterate over the tokens
        for (String token : tokens) {
            // if the token is an operand, push it to the stack
            if (isOperand(token)) {
                operands.push(Integer.parseInt(token));
            }
            // if the token is an operator, push it to the stack
            else if (isOperator(token)) {
                operators.push(token);
            }
            // if the token is a left parenthesis, push it to the stack
            else if (token.equals("(")) {
                operators.push("(");
            }
            // if the token is a right parenthesis, pop the stack until the left parenthesis is reached
            else if (token.equals(")")) {
                while (!operators.isEmpty() && !operators.peek().equals("(")) {
                    int operand2 = operands.pop();
                    int operand1 = operands.pop();
                    String operator = operators.pop();
                    result = apply(operand1, operand2, operator);
                    operands.push(result);
                }
            }
            else {
                // if the token is not an operand or operator, throw an exception
                throw new IllegalArgumentException("Invalid input");
            }

        }

        return result;
    }

    private static int apply(int operand1, int operand2, String operator) {
        return switch (operator) {
            case "plus" -> operand1 + operand2;
            case "minus" -> operand1 - operand2;
            case "times" -> operand1 * operand2;
            case "div" -> operand1 / operand2;
            default -> 0;
        };
    }

    private static boolean isOperator(String token) {
        return token.equals("plus") || token.equals("minus") || token.equals("times") || token.equals("div");
    }

    private static boolean isOperand(String token) {
        try {
            Integer.parseInt(token);
            return true;
        } catch (NumberFormatException e) {
            return false;
        }
    }


    /* test function to test the above function */
    public static void main(String[] args) {

        // test case 1
        String input1 = "( ( 1 plus 5 ) minus 4 )";
        System.out.println("Input: " + input1);
        System.out.println("Output: " + calculate(input1));

        // test case 2
        String input2 = "( ( 1 plus 3 ) times 4 )";
        System.out.println("Input: " + input2);
        System.out.println("Output: " + calculate(input2));

    }
}

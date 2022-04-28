//date: 2022-04-28T17:13:53Z
//url: https://api.github.com/gists/6f5f0b3c76c3f0dd749f088a5b5e5239
//owner: https://api.github.com/users/marciosouzajunior

public class FibonacciNoRecursion {

    public static void main(String[] args) {

        FibonacciNoRecursion fnr = new FibonacciNoRecursion();
        fnr.print(5);

    }

    void print(int n) {

        int first = 0;
        int second = 1;
        int sum;
        System.out.print("0, 1");

        for (int i = 0; i <= n; i++) {
            sum = first + second;
            first = second;
            second = sum;
            System.out.print(", " + sum);
        }

    }

}
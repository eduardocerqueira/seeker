//date: 2021-12-22T17:00:37Z
//url: https://api.github.com/gists/2ea8b27de3b3613b1200bb453bea5fa6
//owner: https://api.github.com/users/Rogal3

import java.math.BigInteger;
import java.util.Scanner;

/**
 * @see https://www.acmicpc.net/problem/2407
 * @author rogal
 */

class Main {
    public static void main(String[]args){
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
        int m = sc.nextInt();
        System.out.println(getCombination(n, m));
    }

    static BigInteger getPermutation(int n, int m) {
        BigInteger result = BigInteger.valueOf(1);
        for (int i = n - m + 1; i <= n; i++) {
            result = result.multiply(BigInteger.valueOf(i));
        }
        return result;
    }

    static BigInteger getFactorial(int n) {
        return getPermutation(n, n);
    }

    static BigInteger getCombination(int n, int m) {
        int alterM = Math.min(m, n - m);
        return getPermutation(n, alterM).divide(getFactorial(alterM));
    }
}

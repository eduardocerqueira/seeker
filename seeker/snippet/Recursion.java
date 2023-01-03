//date: 2023-01-03T16:57:43Z
//url: https://api.github.com/gists/5089533a4c3b74266bfced2ea8cbc1f7
//owner: https://api.github.com/users/profyoni

package edu.touro.cs.mcon364;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class QuickSort {

    public static void quickSort(int[] arr, int low, int high) {
        // check for empty or null array
        if (arr == null || arr.length <= 1) {
            return;
        }

        if (low >= high) {
            return;
        }

        int pivot = arr[high];

        // make left < pivot and right > pivot
        int i = low, j = high-1;
        while (i <= j) {
            // check until all values on left side array are lower than pivot
            while (arr[i] < pivot) {
                i++;
            }
            // check until all values on left side array are greater than pivot
            while (i <= j && arr[j] > pivot) {
                j--;
            }

            // swap values from both side of lists if i is still smaller or equal to j
            if (i <= j) {
                swap(arr, i, j);
                i++;
                j--;
            }
        }
        swap(arr, high, i); //  out pivot in place
        // Do same operation as above recursively to sort two sub arrays
        if (low < i-1) {
            quickSort(arr, low, i-1);
        }
        if (high > i+1) {
            quickSort(arr, i+1, high);
        }
    }

    public static void swap(int[] arr, int i, int j) {
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
}

public class Recursion {

    @Test
    public void  quickSort()
    {
        int[] arr = {9, 8, 7, 6, 5, 4, 3, 2, 1, 11};

// sort the whole array
        QuickSort.quickSort(arr, 0, arr.length - 1);
        for (int i=0; i< arr.length-1; i++)
            assertTrue(arr[i] <= arr[i+1]);
    }

    /***
     * Factorial via Iteration
     * @param n
     * @return
     */
    public static long factorial(int n)
    {
        long t = 1;
        for (int i=1;i<=n;i++)
            t *= i;
        return t;
    }

    /***
     * Factorial via Recursion
     * @param n
     * @return
     */
    public static long factorialR(int n)
    {
        if (n == 0) // Base Case
            return 1;
        long f = factorialR(n - 1);
        return n * f;
    }
    static long[] memo = new long[1000];

    public static long fibonacci(int n)
    {
        if (n ==0 || n == 1)
            return n;
        // if already computed dont do it again
        if (memo[n] != 0)
            return memo[n];
        memo[n] = fibonacci(n-2) + fibonacci(n-1);
        return memo[n];
    }

    @Test
    public void fib1(){
        long x = fibonacci(50);
        assertEquals(12_586_269_025L,x);
    }

    @Test
    public void rec1()
    {
        assertEquals(120, factorial(5));
    }


    @Test
    public void rec2()
    {
        assertEquals(120, factorialR(5));
    }

    public static boolean isPalindromeIt(String s)
    {
        for (int i=0, j=s.length()-1; i < s.length()/2; i++,j--)
        {
            if (s.charAt(i) != s.charAt(j))
                return false;
        }
        return true;
    }

    public static boolean isPalindromeRec(String s) {
        return s.length() < 2 || s.charAt(0) == s.charAt(s.length() - 1) && isPalindromeRec(s.substring(1, s.length() - 1));
    }
    @Test
    public void isPalI()
    {
        assertFalse(isPalindromeIt("HELLO"));
        assertTrue(isPalindromeIt("RACECAR"));
    }
    @Test
    public void isPalR()
    {
        assertFalse(isPalindromeRec("HELLO"));
        assertTrue(isPalindromeRec("RACECAR"));
    }
}

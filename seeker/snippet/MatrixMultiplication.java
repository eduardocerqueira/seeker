//date: 2022-10-11T17:12:33Z
//url: https://api.github.com/gists/cb964a86a0e3cd8011b3cfd0cf154b07
//owner: https://api.github.com/users/Engelshell

/* Multiplication of two matrixes with simple iterative algorithm
 * References: https://en.wikipedia.org/wiki/Matrix_multiplication_algorithm
 * 
 * @author Shelby Engels
 */

public class MatrixMultiplication {

    
    /** 
     * @param C 2D int array to be printed
     * @param name name of array to print
     */
    public static void PrintMatrix(int[][] C, String name) {

        System.out.print(name);
        // iterate over arrays in C
        for (int i = 0; i < C.length; i++) {
            System.out.printf("{ ");
            // iterate over elements in array
            for (int x = 0; x < C[i].length; x++) {
                // don't print last comma in array
                String comma1 = x + 1 < C[i].length ? ", " : " ";
                // print value of element
                System.out.printf("%d%s", C[i][x], comma1);
            }
            // don't print last comma in array
            String comma2 = i + 1 < C.length ? ", " : "";
            System.out.printf("}%s", comma2);
        }
        System.out.println();

    }

    
    /** 
     * @param A A's rows multiply B's columns
     * @param B B multiplied by A
     * @param C fills this as a result
     */
    public static void MultiplyMatrix(int[][] A, int[][] B, int[][] C) {
        // referenced from
        // https://en.wikipedia.org/wiki/Matrix_multiplication_algorithm#Iterative_algorithm

        // iterate N arrays in A
        for (int row = 0; row < A.length; row++) {
            // iterate N elements in B
            for (int col = 0; col < B[0].length; col++) {
                // set to 0 in case we're overwriting a value in C.
                C[row][col] = 0;
                // Limit to N arrays in B
                for (int i = 0; i < B.length; i++) {
                    // Take row of A and multiply column of B
                    // Add result to C
                    C[row][col] += A[row][i] * B[i][col];
                }
            }
        }

    }

    public static void main(String[] args) {

        int A0[][] = { { 2, -1, 0, 1 }, { 1, 0, -1, 2 }, { 0, -1, 1, 0 } };
        int B0[][] = { { 0, 1, -1 }, { 1, -1, 2 }, { -1, 1, 0 }, { 2, 0, -1 } };
        // expected: {1,3,-5} {5,0,-3} {-2,2,-2}
        int A1[][] = { { 8, 2, 7, 0, 5 }, { 3, 4, -1, 2, 6 }, { 0, 2, 1, 8, 7 } };
        int B1[][] = { { 4, 9, -5, 9, 9 }, { 2, 9, 4, 2, 4 }, { -1, 1, 5, 1, 1 }, { 1, 4, -8, -5, 4 },
                { 8, 1, 0, 6, -10 } };
        // expected: { 69,102,3,113,37 } { 71,76,-20,60,-10 } { 67,58,-51,7,-29 }
        int A2[][] = { { 8, -1, 9 }, { 1, -10, 9 }, { 2, -3, 8 } };
        int B2[][] = { { 2 }, { 1 }, { -4 } };
        // expected: {-21, -44, -31}
        int A3[][] = { { 2 }, { 1 }, { 5 } };
        int B3[][] = { { 0, 1, -1 } };
        // expected: {0,2,-2} {0,1,-1} {0,5,-5}
        int A4[][] = { { 9, 4, -2, 9 }, { 4, -7, 89, 200 } };
        int B4[][] = { { 7, 20, 7 }, { 4, 1, -9 }, { -86, 8, 2 }, { 5, 6, 8 } };
        // expected: {296,222,95} {-6654,1985,1869}
        int A5[][] = { { -50, -49, -48 } };
        int B5[][] = { { 1, 2, 3, 4 }, { 9, 8, 7, 6 }, { -1, -5, -2, -7 } };
        // expected: {-443, -252, -397, -158}

        // create result arrays, first level sized to N arrays in A,
        // second level sized to N elements of B.
        int C0[][] = new int[A0.length][B0[0].length];
        int C1[][] = new int[A1.length][B1[0].length];
        int C2[][] = new int[A2.length][B2[0].length];
        int C3[][] = new int[A3.length][B3[0].length];
        int C4[][] = new int[A4.length][B4[0].length];
        int C5[][] = new int[A5.length][B5[0].length];

        // multiply each matrix, could be done in a loop but more complex
        MultiplyMatrix(A0, B0, C0);
        MultiplyMatrix(A1, B1, C1);
        MultiplyMatrix(A2, B2, C2);
        MultiplyMatrix(A3, B3, C3);
        MultiplyMatrix(A4, B4, C4);
        MultiplyMatrix(A5, B5, C5);

        // print each result matrix
        PrintMatrix(C0, "C0");
        PrintMatrix(C1, "C1");
        PrintMatrix(C2, "C2");
        PrintMatrix(C3, "C3");
        PrintMatrix(C4, "C4");
        PrintMatrix(C5, "C5");

    }

}
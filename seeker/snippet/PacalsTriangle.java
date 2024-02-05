//date: 2024-02-05T17:07:39Z
//url: https://api.github.com/gists/38e7f857ef121f81f31f9b6edde2cb46
//owner: https://api.github.com/users/mathewjustin

package com.codepower.striver.arrays.pascals;

import java.util.ArrayList;
import java.util.List;

/**
 * @author Justin Mathew @dev_io
 */
public class PacalsTriangle {
        public static List<List<Integer>> generate(int numRows) {
            List<List<Integer>> triangle = new ArrayList<>();

            // Base case: if no rows are needed, return the empty triangle
            if (numRows == 0) {
                return triangle;
            }

            // First row is always [1]
            triangle.add(new ArrayList<>());
            triangle.get(0).add(1);

            // Generate each row of Pascal's Triangle
            for (int rowNum = 1; rowNum < numRows; rowNum++) {
                List<Integer> row = new ArrayList<>();
                List<Integer> prevRow = triangle.get(rowNum - 1);

                // The first row element is always 1
                row.add(1);

                // Each triangle element (except the first and last of each row) is equal to
                // the sum of the elements above-and-to-the-left and above-and-to-the-right
                for (int j = 1; j < rowNum; j++) {
                    row.add(prevRow.get(j - 1) + prevRow.get(j));
                }

                // The last row element is always 1
                row.add(1);

                triangle.add(row);
            }

            return triangle;
        }

        public static void main(String[] args) {
            int n = 5; // Example to generate 5 rows of Pascal's Triangle
            List<List<Integer>> result = generate(n);

            // Print Pascal's Triangle
            for (List<Integer> row : result) {
                for (int num : row) {
                    System.out.print(num + " ");
                }
                System.out.println();
            }
        }
}

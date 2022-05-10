//date: 2022-05-10T17:14:12Z
//url: https://api.github.com/gists/0f8bb43e3697e8b3d56347e1846e0d4f
//owner: https://api.github.com/users/arc0balen0

package com.khudenko;

public class NewProjectMatrix {
    public static void main(String[] args) {
        int[][] matrix = {{1 , 2, 3},
                          {2, -3, 0},
                          {3, 2, 1}};
        int value = 2;
        boolean found=false;

        for(int i = 0; i < matrix.length; i++) {
            for(int j = 0; j < matrix[i].length; j++) {
                if(matrix[i][j]==value) {
                    found=true;
                    System.out.println("found at index: ["+i+","+j+"]");
                    break;// вложенный цикл прерывается
                }
            }
        }
        if(!found)
            System.out.println("not found");
    }
}

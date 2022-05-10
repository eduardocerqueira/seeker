//date: 2022-05-10T17:18:15Z
//url: https://api.github.com/gists/0980bfffa121a6a310fdca0d76f582f5
//owner: https://api.github.com/users/arc0balen0

package com.khudenko;

public class NewProjectBreak {
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
                    break mark1;// вложенный цикл прерывается
                }
            }
            
        }
        if(!found)
            System.out.println("not found");
    }
}

//date: 2021-12-01T17:19:05Z
//url: https://api.github.com/gists/592d8499cc373d2e7b9310d19d5b01ec
//owner: https://api.github.com/users/mirzacc

class Solution {
    public void setZeroes(int[][] matrix) {
        int col = 0, n = matrix.length, m = matrix[0].length;
        for(int i = 0; i < n; i++){
            if(matrix[i][0] == 0)
                col = 1;
            for(int j = 1; j < m; j++){
                if(matrix[i][j] == 0){
                    matrix[i][0] = matrix[0][j] = 0;
                }
            }
        }
        for(int i = n-1; i >= 0; i--){
            for(int j = m-1; j >= 1; j--){
                if(matrix[i][0] == 0 || matrix[0][j] == 0){
                    matrix[i][j] = 0;
                }
            }
            if(col == 1)
                matrix[i][0] = 0;
        }
    }
}
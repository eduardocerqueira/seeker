//date: 2022-03-30T17:03:45Z
//url: https://api.github.com/gists/e9677682009a4b333013cfb82737f36c
//owner: https://api.github.com/users/black2code

/*
장민욱
*/

public class Quiz01 {
    public static void main(String[] args) {
        System.out.println("[구구단 출력]");
        for (int i = 1; i < 10; i++) {
            for (int j = 1; j < 10; j++) {
                System.out.print(String.format("%02d x  %02d = %02d     ", j, i, j * i));
                if(j==9){
                    System.out.println();
                }
            }
        }
    }
}
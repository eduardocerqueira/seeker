//date: 2024-09-05T17:03:58Z
//url: https://api.github.com/gists/a7e890c666bbee231c2f5169198373f3
//owner: https://api.github.com/users/mu-nss

/*
 *  과제1. 콘솔 화면에 구구단 출력하기
 *  과제 제출자: 문소정
 */

public class Main {
    public static void main(String[] args) {

        String result;

        System.out.println("[구구단 출력]");
        for (int row = 1; row < 10; row++) {
            for (int col = 1; col < 10; col++) {
                result = String.format("%02d x %02d = %02d\t", col, row, row * col);
                System.out.print(result);
            }
            System.out.println();
        }

    } // end of main()
}// end of Main
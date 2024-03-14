//date: 2024-03-14T16:52:36Z
//url: https://api.github.com/gists/a4efaafd1d65f9fd25754d5c7968bf50
//owner: https://api.github.com/users/jjm159

package org.example.assignment_03_18;

/* 정재명
 *
 * 콘솔 화면에 구구단 출력하기
 *
 * JAVA의 다중 반복문, format함수, 제목 및 1단부터 9단까지 표시(예시와 동일 레이아웃)
 */
public class p01 {
    public static void main(String[] args) {

        StringBuilder output = new StringBuilder("[구구단 출력]\n");

        // 반복문
        for (int i = 1; i <= 9; i++) {
            for (int j = 1; j <= 9; j++) {
                // String.format 함수
                String result = String.format("%02d * %02d = %02d\t", j, i, i * j);
                output.append(result);
            }
            output.append("\n");
        }

        System.out.println(output);
    }
}

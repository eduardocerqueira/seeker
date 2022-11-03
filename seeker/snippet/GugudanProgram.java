//date: 2022-11-03T17:06:59Z
//url: https://api.github.com/gists/43c552adb20635da57fc18edcb4924b6
//owner: https://api.github.com/users/sunsik17

package gugudanprogram;
/*
남순식

String.fomat 함수를 활용하여 1의자리 숫자를 빈자리에 0을 붙혀 두자리수로 표현
String.fomat 은 문자열을 하나로 통일 할 수 있다.

출력할때 구구단을 세로로 만들기 위해 j * i 모양으로 출력
*/               

public class GugudanProgram {
    public static void main(String[] args) {

        for (int i = 1; i < 10; i++) {
            String space = "    ";
            System.out.println();
            for (int j = 1; j < 10; j++) {
                if (j == 9) {
                    space = "";
                }
                System.out.printf(String.format("%02d", j) + " x " + String.format("%02d", i) + " = " + String.format("%02d", i * j) + space);       
            }
        }
    }
}

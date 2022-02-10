//date: 2022-02-10T16:43:40Z
//url: https://api.github.com/gists/ee646da16066f5e1392741b83f524e70
//owner: https://api.github.com/users/lismsdh2

import java.util.Scanner;

public class Q2 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in) ;
        boolean isRun = true;

        while(isRun){

            System.out.println("1. 회원 관리");
            System.out.println("2. 과목 관리");
            System.out.println("3. 수강 관리");
            System.out.println("4. 결제 관리");
            System.out.println("5. 종료");
            int inputNum = sc.nextInt();
    
            switch(inputNum){
                case 1:
                    System.out.println("회원 관리 메뉴를 선택했습니다.");
                    break;
                case 2:
                    System.out.println("과목 관리 메뉴를 선택했습니다.");
                    break;
                case 3:
                    System.out.println("수강 관리 메뉴를 선택했습니다.");
                    break;
                case 4:
                    System.out.println("결제 관리 메뉴를 선택했습니다.");
                    break;
                case 5:
                    System.out.println("프로그램을 종료 합니다.");
                    isRun = false;
                    break;
            }
        }
        sc.close();
    }
}

//date: 2024-06-14T16:45:05Z
//url: https://api.github.com/gists/f0398d138b524743b86cfd41b662eff3
//owner: https://api.github.com/users/kej9946

/*
김은지 | 주민등록번호 생성 프로그램
*/

import java.util.Scanner;

public class javaMini04 {
    public static void main(String[] args) {

        System.out.println("[주민등록번호 계산]");
        Scanner scanner = new Scanner(System.in);

        System.out.print("출생년도를 입력해 주세요.(yyyy):");
        int birthYear = scanner.nextInt();
        System.out.print("출생월을 입력해 주세요.(mm):");
        int birthMonth = scanner.nextInt();
        System.out.print("출생일을 입력해 주세요.(dd):");
        int birthDay = scanner.nextInt();
        System.out.print("성별을 입력해 주세요.(m/f):");
        char gender = scanner.next().charAt(0);

        int genderNum = 0;
        if (gender == 'm'){
            genderNum = 3;
        }else if (gender == 'f'){
            genderNum = 4;
        }else{
            System.out.println("올바른 성별을 입력해 주세요");
            scanner.close();
            return;
        }

        String frontNum = String.format("%02d%02d%02d", birthYear % 100, birthMonth, birthDay);

        int randomBackNum = (int) (Math.random() * 999999) + 1;
        String backNum = String.format("%d%06d", genderNum, randomBackNum);

        String idNumber = frontNum + "-" + backNum;
        System.out.println(idNumber);
        
        Scanner.close();

    }

}
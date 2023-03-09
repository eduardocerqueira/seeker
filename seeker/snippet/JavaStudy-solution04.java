//date: 2023-03-09T17:07:30Z
//url: https://api.github.com/gists/e54a5178a7b3016e2e8622878a4c7245
//owner: https://api.github.com/users/dev-Seonghwan

import java.util.Random;
import java.util.Scanner;

public class solution {
    public static void main(String[] args) {
            
            Scanner sc = new Scanner(System.in);
            Random rd = new Random();

            String sex;

            int birthYear;
            int birthMonth;
            int birthDate;

            System.out.println("[주민등록번호 계산 ]");

            System.out.println("출생년도를 입력해 주세요.(yyyy)");
            birthYear = sc.nextInt();

            System.out.println("출생월을 입력해 주세요.(mm)");
            birthMonth = sc.nextInt();

            System.out.println("출생일을 입력해 주세요.(dd)");
            birthDate = sc.nextInt();

            System.out.println("성별을 입력해 주세요.(m/f)");
            sex = sc.next();

            StringBuilder sb = new StringBuilder();

            sb.append(Integer.toString(birthYear).substring(2));
            sb.append(String.format("%02d", birthMonth));
            sb.append(String.format("%02d", birthDate));
            sb.append("-");

            if (sex.equals("m")) {
                if (birthYear < 2000) {
                    sb.append("1");
                } else if (birthYear >= 2000) {
                    sb.append("3");
                }
            } else {
                if (birthYear < 2000) {
                    sb.append("2");
                } else if (birthYear >= 2000) {
                    sb.append("4");
                }
            }
            sb.append(rd.nextInt(999999));

            System.out.println(sb); //result
        
    }
}

//date: 2024-03-14T16:57:44Z
//url: https://api.github.com/gists/1cee4068a893f73a500f2df25a33d51c
//owner: https://api.github.com/users/jjm159

package org.example.assignment_03_18;

/* 정재명
* 주민등록번호 생성 프로그램
*
* 1. 주민등록번호 생성 로직에 맞게 주민등록번호 생성
* 2. 입력값은 생년, 월, 일, 성별과 임의의 번호를 통해서 생성
* 3. 임의번호는 Random함수의 nextInt()함수를 통해서 생성
*    (임의 번호 범위는 1 ~ 999999사이의 값으로 설정)
* */

import java.util.Random;
import java.util.Scanner;

public class p04 {
    static public void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        System.out.print("출생년도를 입력해 주세요.(yyyy):");
        int year = sc.nextInt();
        sc.nextLine();

        System.out.print("출생월을 입력해 주세요.(mm):");
        int month = sc.nextInt();
        sc.nextLine();

        System.out.print("출생일을 입력해 주세요.(dd):");
        int day = sc.nextInt();
        sc.nextLine();

        System.out.print("성별을 입력해 주세요.(m/f):");
        String gen = sc.nextLine();
        int genNum = gen.equals("m") ? 3:4;

        Random random = new Random();

        int ranNum = random.nextInt(1, 999999);

        String result = String.format(
                "%02d%02d%02d-%d%06d",
                year % 100,
                month,
                day,
                genNum,
                ranNum
        );
        System.out.println(result);
    }
}

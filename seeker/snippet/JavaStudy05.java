//date: 2022-07-01T17:05:50Z
//url: https://api.github.com/gists/aef6815dd04efdb00895c0479a6028c8
//owner: https://api.github.com/users/Chanssori

/*
오진석
*/

import java.util.Scanner;
import java.time.LocalDate;
import java.time.DayOfWeek;

public class JavaStudy05 {

    public static void main(String[] args) {

        Scanner sc = new Scanner(System.in);

        System.out.println("[달력 출력 프로그램]");

        System.out.print("달력의 년도를 입력해 주세요.(yyyy):");
        int year = sc.nextInt();

        System.out.print("달력의 월을 입력해 주세요.(mm):");
        int month = sc.nextInt();

        System.out.println();
        System.out.println();


        LocalDate date = LocalDate.of(year, month, 1);

        // 1일의 요일
        DayOfWeek dayOfFirst = date.getDayOfWeek();
        int dayOfFirstIndex = dayOfFirst.getValue();

        // 달의 마지막 날
        int dayOfLast = date.lengthOfMonth();

        // 맨위 부분
        System.out.println(String.format("[%d년 %d월]", year, month));
        String arr[] = {"일","월","화","수","목","금","토"};
        for(int i=0; i<7;i++){
            System.out.print(arr[i]+"\t");
        }
        System.out.println();

        for(int i=0;i<dayOfFirstIndex;i++){
            System.out.print("\t");
        }

        // 날짜
        int total = dayOfFirstIndex;
        for(int i=1;i<dayOfLast+1;i++){
            if(total==7) {
                System.out.println();
                total = 0;
            }
            System.out.print(String.format("%02d\t",i));
            total++;
        }
    }
}


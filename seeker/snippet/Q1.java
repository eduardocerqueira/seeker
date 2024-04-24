//date: 2024-04-24T16:48:14Z
//url: https://api.github.com/gists/ddc288f7c9d722ae264c0068368fb985
//owner: https://api.github.com/users/GahyungKim

import java.time.LocalDate;
import java.util.Scanner;

public class Q1 {

    public static int dayCheck(int day){
        if(day == 7){
            return 0;
        }
        else{
            return day;
        }
    }

    public static void makeCal(int year, int month) {
        LocalDate calDay1 = LocalDate.of(year, month - 1, 1);
        LocalDate calDay2 = LocalDate.of(year, month, 1);
        LocalDate calDay3 = LocalDate.of(year, month + 1, 1);

        int day1 = calDay1.getDayOfWeek().getValue(); //5
        int day2 = calDay2.getDayOfWeek().getValue(); //7
        int day3 = calDay3.getDayOfWeek().getValue(); //3

        day1 = dayCheck(day1);
        day2 = dayCheck(day2);
        day3 = dayCheck(day3);

        int lastDay1 = calDay1.lengthOfMonth();
        int lastDay2 = calDay2.lengthOfMonth();
        int lastDay3 = calDay3.lengthOfMonth();

        String yearStr = Integer.toString(year);
        String monthStr1 = String.format("%02d", month - 1);
        String monthStr2 = String.format("%02d", month);
        String monthStr3 = String.format("%02d", month + 1);

        System.out.println("[" + yearStr + "년 " + monthStr1 + "월]\t\t\t\t\t\t[" + yearStr + "년 " + monthStr2 + "월]\t\t\t\t\t\t[" + yearStr + "년" + monthStr3 + "월]");
        System.out.println("일\t월\t화\t수\t목\t금\t토\t\t\t일\t월\t화\t수\t목\t금\t토\t\t\t일\t월\t화\t수\t목\t금\t토");

        int date1 = 1;
        int date2 = 1;
        int date3 = 1;

        for (int i = 0; i < 7; i++) {
            if (i < day1) {
                System.out.print("\t");
            } else {
                System.out.printf("%02d\t", date1);
                date1++;
            }
        }
        System.out.print("\t\t");

        for (int i = 0; i < 7; i++) {
            if (i < day2) {
                System.out.print("\t");
            } else {
                System.out.printf("%02d\t", date2);
                date2++;
            }
        }
        System.out.print("\t\t");

        for (int i = 0; i < 7; i++) {
            if (i < day3) {
                System.out.print("\t");
            } else {
                System.out.printf("%02d\t", date3);
                date3++;
            }
        }
        System.out.println();

        day1 = 0;
        day2 = 0;
        day3 = 0;
        while (date1 < lastDay1 || date2 < lastDay2 || date3 < lastDay3) {
            for (int i = 0; i < 7 && date1 <= lastDay1; i++) {
                System.out.printf("%02d\t", date1);
                date1++;
                day1++;
            }
            System.out.print("\t\t");

            if (day1 % 7 != 0 && date1 == lastDay1 + 1) {
                for (int i = 0; i < 7 - (day1 % 7); i++) {
                    System.out.println("\t");
                }
            }
            for (int i = 0; i < 7 && date2 <= lastDay2; i++) {
                System.out.printf("%02d\t", date2);
                date2++;
                day2++;
            }
            System.out.print("\t\t");

            if (day2 % 7 != 0 && date2 == lastDay2 + 1) {
                for (int i = 0; i < 7 - (day2 % 7); i++) {
                    System.out.print("\t");
                }
            }
            for (int i = 0; i < 7 && date3 <= lastDay3; i++) {
                System.out.printf("%02d\t", date3);
                date3++;
                day3++;
            }
            System.out.println();

        }
    }

    public static void main(String[] args){
        Scanner sc = new Scanner(System.in);
        System.out.println("[달력 출력 프로그램]");
        System.out.print("달력의 년도를 입력해 주세요. (yyyy):");
        int year = sc.nextInt();
        System.out.print("달력의 월을 입력해 주세요.(mm):");
        int month = sc.nextInt();

        makeCal(year, month);

    }
}

//date: 2023-08-03T16:45:37Z
//url: https://api.github.com/gists/4476f0a4ab559eff672e80b81d31b021
//owner: https://api.github.com/users/JGoo99

/*
구진
미니과제 05 : 달력 출력 프로그램 (배점 10점)
*/

import java.time.LocalDate;
import java.util.Scanner;

public class MiniAsg_05 {
  public static void main(String[] args) {

    System.out.println("[달력 출력 프로그램]");

    MyCalendar calendar = new MyCalendar();
    calendar.getCalender();
  }
}

class MyCalendar {
  private int year;
  private int month;
  void getCalender() {
    Scanner sc = new Scanner(System.in);

    System.out.print("달력의 년도를 입력해 주세요.(yyyy):");
    this.year = sc.nextInt();

    System.out.print("달력의 월을 입력해 주세요.(mm):");
    this.month = sc.nextInt();

    System.out.println("\n");

    printCalendarForm();
  }

  void printCalendarForm() {
    // 년, 월, 요일 출력
    System.out.format("[%d년 %d월]%n", year, month);
    String[] week = {"일", "월", "화", "수", "목", "금", "토"};
    for (int i = 0; i < week.length; i++) {
      System.out.printf("%-2s\t", week[i]);
    }
    System.out.println();

    printCalendarDay();
  }

  void printCalendarDay() {
    LocalDate date = LocalDate.of(year, month, 1);

    int n = 1; // 날짜 (01, 02, 03, ...)
    int weekIndex = date.getDayOfWeek().getValue();
    // 만약 7로 초기화된 채 반복문을 돌면 달력의 첫줄이 공백처리됨
    if (weekIndex == 7) weekIndex = 0;

    // 날짜 배열 초기화
    int[][] day = new int[6][7];
    for (int i = 0; i < day.length; i++) {
      for (int j = weekIndex; j < day[i].length; j++) {
        day[i][j] = n++;
      }
      weekIndex = 0;
    }

    // 날짜 배열 출력
    for (int i = 0; i < day.length; i++) {
      for (int j = 0; j < day[i].length; j++) {

        // 00일은 공백으로 출력함
        if (day[i][j] == 0) {
          System.out.print("  \t");

        // 해당 달의 마지막 날까지만 출력하고 반복문 중지
        } else if (day[i][j] <= date.lengthOfMonth()) {
          System.out.printf("%02d\t", day[i][j]);
        } else {
          break;
        }
      }
      System.out.println();
    }
  }
}
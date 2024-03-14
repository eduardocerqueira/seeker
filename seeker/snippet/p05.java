//date: 2024-03-14T16:58:17Z
//url: https://api.github.com/gists/1139efc6be10f80bf8bd431de166659d
//owner: https://api.github.com/users/jjm159

package org.example.assignment_03_18;

/* 정재명
*
* 달력 출력 프로그램
* 1. 입력받은 년도와 월을 통한 달력 생성
* 2. 입력값은 년도, 월을 입력
* 3. 날짜는 LocalDate클래스를 이용(Calendar와 Date클래스도 이용 가능)
* 4. 출력은 입력한 달을 기준으로 이전달, 입력달, 현재달 출력(3달 출력)
*
* */

import java.util.Arrays;
import java.util.Calendar;
import java.util.List;
import java.util.Scanner;
import java.util.stream.Collectors;

public class p05 {

    public static void main(String[] args) {

        Scanner sc = new Scanner(System.in);

        System.out.println("[달력 출력 프로그램]");

        System.out.print("달력의 년도를 입력해 주세요.(yyyy):");
        int year = sc.nextInt();
        sc.nextLine();

        System.out.print("달력의 월을 입력해주세요.(mm):");
        int month = sc.nextInt();
        sc.nextLine();

        String[] prev = getCalendarString(year - 1, (month - 1 - 1 + 12) % 12 + 1);
        String[] curr = getCalendarString(year, month);
        String[] next = getCalendarString(year + 1, (month + 1 - 1 + 12) % 12 + 1);

        prev = makePrintableString(0, prev);
        curr = makePrintableString(1, curr);
        next = makePrintableString(2, next);

        String result = makeTotalResult(new String[][]{prev, curr, next});
        System.out.println(result);
    }

    private static String[] getCalendarString(int year, int month) {
        int titleCount = 1;
        int dayOfWeekLineCount = 1;
        int maxDayLineCount = 6;
        String[] result = new String[titleCount + dayOfWeekLineCount + maxDayLineCount];
        Arrays.fill(result, "\t".repeat(7));

        // title
        String title = String.format(
                "[%d년 %02d월]\t\t\t\t",
                year,
                month
        );
        result[0] = title;

        // day of weeks title
        List<String> dayOfWeeksTitleList = List.of("일", "월", "화", "수", "목", "금", "토");
        String dayOfWeeksTitle = dayOfWeeksTitleList.stream()
                .map(d -> String.format("%s\t", d))
                .collect(Collectors.joining());
        result[1] = dayOfWeeksTitle;

        // day
        Calendar calendar = Calendar.getInstance();
        calendar.set(Calendar.YEAR, year);
        calendar.set(Calendar.MONTH, month - 1);

        int maxMonthDayCount = calendar.getActualMaximum(Calendar.DAY_OF_MONTH);

        int lineIndex = dayOfWeekLineCount + titleCount;
        StringBuilder currentLine = new StringBuilder();

        for (int day = 1; day <= maxMonthDayCount; day++) {
            calendar.set(Calendar.DAY_OF_MONTH, day);

            int dayOfWeek = calendar.get(Calendar.DAY_OF_WEEK);

            String currentDayString = String.format("%02d\t", day);

            // 1일 때 - 해당 요일 이전 요일 개수 만큼 빈칸
            if (day == 1) {
                int emptyCount = dayOfWeek - 1;
                String emptyString = "\t".repeat(emptyCount);
                currentLine.append(emptyString);
            }

            currentLine.append(currentDayString);

            // max day 일 때 - 마지막 전부 빈 스트링
            if (day == maxMonthDayCount) {
                int emptyCount = 7 - dayOfWeek;
                String emptyString = "\t".repeat(emptyCount);
                currentLine.append(emptyString);
            }

            // currenLine이 꽉 찼을때 또는 토요일일 때 - lineIndex에 string 할당
            if (dayOfWeek == 7 || day == maxMonthDayCount) {
                result[lineIndex] = currentLine.toString();
                lineIndex++;
                currentLine = new StringBuilder();
            }
        }

        return result;
    }


    private static String[] makePrintableString(int index, String[] targetStringList) {
        if (index == 0) return targetStringList;

        String tab = "\t\t";

        String[] result = new String[targetStringList.length];

        for (int i = 0; i < targetStringList.length; i++) {
            result[i] = tab + targetStringList[i];
        }

        return result;
    }

    private static String makeTotalResult(String[][] resultList) {
        StringBuilder result = new StringBuilder();

        for (int j = 0; j < 8; j++) {
            StringBuilder currentLine = new StringBuilder();
            for (String[] strings : resultList) {
                currentLine.append(strings[j]);
            }
            if (!currentLine.isEmpty()) {
                result.append(currentLine.toString());
                result.append("\n");
            }
        }

        return result.toString();
    }

}
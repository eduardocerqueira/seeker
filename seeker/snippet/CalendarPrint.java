//date: 2024-02-01T16:49:42Z
//url: https://api.github.com/gists/4b5fefe344d9d64d515f75c6cff42eb2
//owner: https://api.github.com/users/DNGHKM

/*
    김동하
    날짜 관련 클래스들을 처음 다루다 보니 
    익숙치 않아 구현하기 어려웠습니다.
    더 좋은 방법이 있는지 궁금합니다.
 */
import java.time.LocalDate;
import java.util.Scanner;

public class CalendarPrint {
    public static void main(String[] args) {
        String[] DayOfWeek={"일","월", "화", "수", "목","금","토"};
        Scanner sc = new Scanner(System.in);
        System.out.println("[달력 출력 프로그램]");
        System.out.print("달력의 년도를 입력해 주세요.(yyyy): ");
        int year = sc.nextInt();
        System.out.print("달력의 월을 입력해 주세요.(mm): ");
        int month = sc.nextInt();
        LocalDate now = LocalDate.of(year,month,1);
        LocalDate before = now.minusMonths(1);
        LocalDate next = now.plusMonths(1);
        LocalDate[] Months = {before, now, next};
        boolean[] finish = new boolean[3];

        //달력 상단 년,월 출력
        for (int i = 0; i <3; i++) {
            if(i<2) {
                System.out.print("[" + Months[i].getYear() + "년 " + String.format("%02d",Months[i].getMonthValue()) + "월]\t\t\t\t\t\t");
            }else{
                System.out.print("[" + Months[i].getYear() + "년 " + String.format("%02d",Months[i].getMonthValue()) + "월]\n");
            }
        }

        //요일 출력
        for (int i = 0; i <3; i++) {
            for (String s : DayOfWeek) {
                System.out.print(s+"\t");
            }
            System.out.print("\t\t");
        }
        System.out.println();
        int[] days = {1, 1, 1};

        //일(첫줄)출력
        for (int i = 0; i <3; i++) {
            while (true) {
                int count = Months[i].getDayOfWeek().getValue();
                //1을 일요일, 6을 토요일로 바꿈
                if(count<7){
                    count++;
                }else{
                    count-=6;
                }
                //월요일에서 시작하지 않으면 앞에 공백 출력
                for (int j = 0; j <(count-1); j++) {
                    System.out.print("\t");
                }
                while(true) {
                    System.out.print(String.format("%02d",days[i])+"  ");
                    days[i]++;
                    Months[i] = Months[i].plusDays(1);
                    //각 줄이 토요일까지 출력했으면 break 하고 다음달력 출력
                    if(Months[i].getDayOfWeek().getValue()==7){
                        break;
                    }
                }
                break;
            }
            System.out.print("\t\t");
        }
        System.out.println();

        //나머지 줄 출력
        while(true) {
            for (int i = 0; i < 3; i++) {
                if (days[i] <= Months[i].getDayOfMonth()) {
                    while (true) {
                        while (true) {
                            System.out.print(String.format("%02d", days[i]) + "  ");
                            days[i]++;
                            if (days[i] > Months[i].lengthOfMonth()) {
                                int count = Months[i].getDayOfWeek().getValue();
                                //1을 일요일, 6을 토요일로 바꿈
                                if (count < 7) {
                                    count++;
                                } else {
                                    count -= 6;
                                }
                                //토요일에서 끝나지 않으면 뒤에 공백 출력
                                if (count < 7) {
                                    for (int j = 0; j < (7 - count); j++) {
                                        System.out.print("\t");
                                    }
                                }
                                break;
                            }
                            Months[i] = Months[i].plusDays(1);
                            //각 줄이 토요일까지 출력했으면 break 하고 다음달력 출력
                            if (Months[i].getDayOfWeek().getValue() == 7) {
                                break;
                            }
                        }
                        System.out.print("\t\t");
                        break;
                    }
                }else{
                    //마지막 출력할 날이 없는 달은 줄 밀어내기
                    System.out.print("\t\t\t\t\t\t\t\t\t");
                }
            }
            System.out.println();
            //각 달력이 다 출력되었으면 finish를 true 처리
            for (int i = 0; i < 3; i++) {
                if(days[i]>Months[i].lengthOfMonth()) finish[i]=true;
            }
            //3개 달력이 모두 출력되었으면 프로그램 종료
            if(finish[0]&&finish[1]&&finish[2]) return;
        }
    }
}
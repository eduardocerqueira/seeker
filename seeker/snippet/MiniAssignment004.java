//date: 2023-01-02T16:46:30Z
//url: https://api.github.com/gists/234c0bd657324304c691e128c5a011be
//owner: https://api.github.com/users/wsh096

//미니과제004 주민등록번호 계산_왕석현
//Random 사용한 방법
/*
0~9까지의 숫자를 만드는 random난수를 하나 만들고
이를 6번 반복하며 또한, 이를 클래스에 선언한 빈문자열에 저장한다.
이때, 내부의 ran은 for문을 반복하며 그 때마다 새롭게 값을 형성한다.

각 값을 Integer.toString(rand.nextInt(10));을 해서 String으로 변환했고
이를 numStr에 추가해줌으로써 순서대로 난수를 만들었다.
(역순으로 만드는 방법은 numStr = ran + numStr;

이후는 Scanner로 이전의 3번 문제와 유사하며,
여기서는 연도의 2020년 전후를 기준으로 genderNum을 변경하게 했다.
year값은 100을 나눈 나머지 값으로 복합대입연산자를 써서 마지막에 적절한 숫자로 만들었고
한자리수가 되는 것을 방지하고자 %d가 아닌 %02d를 사용했다.

printf에서 %s를 통해 String 타입의 변수를 받을 수 있도록 해서 문제를 마무리했다.

*/
import java.util.Scanner;
import java.util.Random;

class MiniAssignment004 {
  public static void main(String[] args) {
    System.out.println("[주민등록번호 계산]");
    
    Scanner sc = new Scanner(System.in);
    
    Random rand = new Random();
    String numStr = "";

    for(int i=0;i<6;i++) {
            
    String ran = Integer.toString(rand.nextInt(10));
    numStr += ran;      
            }
    
    int genderNum;
    System.out.print("출생년도를 입력해 주세요.(yyyy): ");
    int year = sc.nextInt();
    System.out.print("출생월을 입력해 주세요.(mm): ");
    int month = sc.nextInt();
    System.out.print("출생일을 입력해 주세요.(dd): ");
    int day = sc.nextInt();
    System.out.print("성별을 입력해 주세요.(m/f): ");
    char gender = sc.next().charAt(0);
    
    if(year<2020){
      if(gender == 'm')
        genderNum = 1;
      else
        genderNum = 2;
    }
    else{
      if(gender == 'm')
        genderNum = 3;
      else
        genderNum = 4;
    }

    year %= 100;
    
      System.out.printf("%02d%02d%02d - %d%s",year,month,day,genderNum,numStr);
    sc.close();
  }
}
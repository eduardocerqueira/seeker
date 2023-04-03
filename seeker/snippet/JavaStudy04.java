//date: 2023-04-03T17:09:43Z
//url: https://api.github.com/gists/5377c5e2afe868ca53c6eab0f4032ccc
//owner: https://api.github.com/users/kny5579

//김나연 과제04
import java.util.*;

public class Practice {

    public static void main(String[] args) {
    	//scanner, 조건문, random클래스
    	int g=0;
    	String answer="";
    	
    	Random random=new Random();
    	Scanner sc=new Scanner(System.in);
    	System.out.println("[주민등록번호 계산]");
    	System.out.print("출생년도를 입력해 주세요.(yyyy):");
    	String year=sc.next();
    	System.out.print("출생월을 입력해 주세요.(mm):");
    	String month=sc.next();
    	System.out.print("출생일을 입력해 주세요.(dd):");
    	String date=sc.next();
    	System.out.print("성별을 입력해 주세요.(m/f):");
    	String gender=sc.next();
    	
    	if(gender.equals("m")) g=3;
    	else if(gender.equals("f")) g=4;
    	
    	for(int i=0;i<6;i++) {
    		answer+=random.nextInt(10);
    	}
    	System.out.println(year+month+date+"-"+g+answer);
    }
}
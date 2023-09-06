//date: 2023-09-06T17:07:27Z
//url: https://api.github.com/gists/87ec6788e385db9e2e28a76856e3e20e
//owner: https://api.github.com/users/park-sang-yong

/*
  박상용
*/
import java.io.*;
import java.util.StringTokenizer;

public class sudden2 {
    public static void main(String[] args) throws IOException {


        System.out.println("나의 x좌표와 y좌표를 입력하세요.ex)1,2");
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        StringTokenizer st = "**********"
        BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(System.out));

        int[] x = new int[11];
        int[] y = new int[11];

        for (int i = 0; i < 11; i++) {
            String str;
            if (i==0){
                str = "**********"
                System.out.println("10개의 좌표를 입력하세요.");

            } else {
                System.out.println(i+"번째 x좌표와 y좌표를 입력하세요.");
                st = "**********"
                str = "**********"
            }
            x[i] = Integer.parseInt(str.substring(0,str.indexOf(',')));
            y[i] = Integer.parseInt(str.substring(str.indexOf(',')+1, str.length()));

            for (int j = 1; j < i; j++) {
                if (x[i] == x[j] && y[i] == y[j]){
                    i--;
                    System.out.print("이미 입력한 좌표입니다.\n다시 ");
                    break;
                }
            }
        }
        int min = Integer.MAX_VALUE;
        int minx = 0;
        int miny = 0;
        for (int i = 1; i <= 10; i++) {
            double dif = Math.pow(Math.abs(x[0] - x[i]),2) + Math.pow(Math.abs(y[0] - y[i]),2);
            if ((double)min > dif){
                min = (int)dif;
                minx = x[i];
                miny = y[i];
            }
        }
        System.out.println("가장 가까운 좌표는 ("+minx+","+miny+") 입니다.");
    }
} 좌표는 ("+minx+","+miny+") 입니다.");
    }
}
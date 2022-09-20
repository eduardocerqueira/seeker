//date: 2022-09-20T17:03:46Z
//url: https://api.github.com/gists/47c00cf478014e9ba680b2cedfc49c5f
//owner: https://api.github.com/users/wjdwlfk12

import java.util.Scanner;
import static java.lang.Integer.parseInt;

public class JavaPractice02{

    public static boolean checkExistsAlready(int[][] arr, int x, int y, int cnt){

        if(cnt == 0){
            return false;
        }


        for(int i = 0; i < cnt; i++){
            if(x == arr[i][0]){
                if(y == arr[i][1]){
                    return true;
                }
            }
        }

        return false;
    }

    public static int[] minAbsValue(int[] myValue, int[] nearestValue, int[] arr){

        int tmpNear = Math.abs(myValue[0] - nearestValue[0]) + Math.abs(myValue[1] - nearestValue[1]);
        int tmpArr = Math.abs(myValue[0] - arr[0]) + Math.abs(myValue[1] - arr[1]);

        if(tmpNear <= tmpArr){
            return nearestValue;
        }else{
            return arr;
        }

    }

    public static void main(String args[]){


        int[] myValue = new int[2];
        int[] nearestValue = new int[2];
        int[][] scValue = new int[10][2];
        int cnt = 0;

        try {


            Scanner sc = new Scanner(System.in);

            System.out.print("당신의 x값을 입력하세요.");
            myValue[0] = parseInt(sc.nextLine());

            System.out.print("당신의 y값을 입력하세요.");
            myValue[1] = parseInt(sc.nextLine());

            while(cnt < 10){
                int x;
                int y;

                System.out.print("비교할 " + (cnt+1) + "번째 x값을 입력하세요.");
                x = parseInt(sc.nextLine());
                System.out.print("비교할 " + (cnt+1) + "번째 y값을 입력하세요.");
                y = parseInt(sc.nextLine());

                if(false == checkExistsAlready(scValue, x, y, cnt)){
                    scValue[cnt][0] = x;
                    scValue[cnt][1] = y;

                    if(cnt == 0){
                        nearestValue = scValue[cnt];
                        cnt++;
                    }else{
                        nearestValue = minAbsValue(myValue, nearestValue, scValue[cnt]);
                        cnt++;
                    }

                }else{
                    System.out.println("값이 중복되었습니다 다시 입력하세요");
                }
            }

            sc.close();

            System.out.print("가장 가까운 값 : ");
            System.out.println("(" + nearestValue[0] +", " + nearestValue[1] + ")");






        }catch (NumberFormatException e){
            System.out.println("정수를 입력하세요");
        }

    }
}
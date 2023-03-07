//date: 2023-03-07T16:53:35Z
//url: https://api.github.com/gists/72bd5e977fe47d385fcedff72be316be
//owner: https://api.github.com/users/jiyoungzero

/* 이지영 */

// 로또 번호가 중복되지 않기 때문에 set으로 받는게 더 나을 것 같다는 생각이 들었습니다.
// set으로 하면 contains를 쓸 수 있어서 굳이 for-if로 판정하는 단계가 없어도 되는 것였습니다. 
// main에서 깔끔하게 함수만 쓰고 나머지 기능은 최대한 함수화했습니다. 

import java.util.*;

public class Problem7 {
    public static int[][] myLotto(int[][] arr){
        Random random = new Random();
        // 랜덤
        for(int i=0;i<arr.length;i++){
            for(int j=0;j<6;j++){
                arr[i][j] = random.nextInt(45)+1;
                for(int k=0;k<j;k++){
                    if(arr[i][k] == arr[i][j]){j--;}
                }
            }
        }

        // 출력
        for(int i=0;i<arr.length;i++){
            char ch = (char)(65+i);
            System.out.print(String.format("%c\t",ch));
            for(int j=0;j<6;j++){
                if (j==5){
                    System.out.print(String.format("%02d ",arr[i][j]));
                }else{
                    System.out.print(String.format("%02d, ",arr[i][j]));
                }
            }
            System.out.println();
        }
        return arr;
    }
    public static int[] announceLotto(){
        System.out.println("[로또 발표]");
        Random random = new Random();
        System.out.print("\t");
        int[] arr = new int[6];
        //랜덤
        for(int i=0;i<6;i++){
            arr[i] = random.nextInt(45)+1;
            for(int j=0;j<i;j++){
                if(arr[i] == arr[j]){i--;}
            }
        }

        // 출력
        for(int i=0;i<5;i++){
            System.out.print(String.format("%02d, ", arr[i]));
        }
        System.out.print(String.format("%02d",  arr[5]));
        return arr;
    }

    public static void myLottoResult(int[][] arr, int[] lotto){
        System.out.println("[내 로또 결과]");
        int[] match = new int[arr.length];

        // 몇개 일치하는지
        for(int i=0;i<arr.length;i++){
            int tmp = 0;
            for(int value = 0;value<6;value++){
                for(int j=0;j<6;j++){
                    if(lotto[value] == arr[i][j]){tmp+=1;}
                }
            }
            match[i] = tmp;
        }

        for(int i=0;i<arr.length;i++){
            char ch = (char)(65+i);
            System.out.print(String.format("%c\t",ch));
            for(int j=0;j<6;j++){
                if (j==5){
                    System.out.print(String.format("%02d => %d개 일치 ",arr[i][j], match[i]));
                }else{
                    System.out.print(String.format("%02d, ",arr[i][j]));
                }
            }
            System.out.println();
        }

    }

    public static void main(String[] args) {
        System.out.println("[로또 당첨 프로그램]");
        System.out.println();

        Scanner sc = new Scanner(System.in);
        System.out.print("로또 개수를 입력해 주세요.(숫자 1~10):");
        int numOfLotto = sc.nextInt();
        int[][] myArray = new int[numOfLotto][6];
        int[] lottoNum = new int[6];

        myArray = myLotto(myArray);
        System.out.println();

        lottoNum = announceLotto();
        System.out.println();
        System.out.println();

        myLottoResult(myArray, lottoNum);

    }
}
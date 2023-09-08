//date: 2023-09-08T16:40:42Z
//url: https://api.github.com/gists/e553c22e0d548b15499afb1f323bf6c8
//owner: https://api.github.com/users/cjwon0827

import java.util.Random;
import java.util.Scanner;

public class project6 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        Random random = new Random();
        int max = -1;
        int idx = 0;

        System.out.print("총 진행할 투표수를 입력해주세요.");
        int totalVote = sc.nextInt();

        System.out.print("가상 선거를 진행할 후보자 인원을 선택해 주세요.");
        int people = sc.nextInt();
        String[] peopleArr = new String[people];
        int[] countVote = new int[people];

        for(int i = 0; i < peopleArr.length; i++){
            System.out.print(i+1 + "번째 후보자 이름을 입력해 주세요.");
            peopleArr[i] = sc.next();
        }
        System.out.println();

        for(int i = 0; i < totalVote; i++){
            int randomNum = random.nextInt(people);
            countVote[randomNum]++;
            System.out.println("[투표진행률]: " + String.format("%.2f", (i+1)/(double)totalVote * 100) + "%, " + (i+1) + "명 투표 => " + peopleArr[randomNum]);
            for(int j= 0; j < people; j++){
                System.out.println("[기호:" + (j+1) + "] " + peopleArr[j] + ":\t" + String.format("%.2f", countVote[j]/(double)totalVote * 100) + "%\t" + "(투표수: " + countVote[j] + ")");
            }
            System.out.println();
        }

        for(int i = 0; i < countVote.length; i++){
            if(max < countVote[i]){
                max = countVote[i];
                idx = i;
            }
        }
        System.out.println("[투표결과] 당선인 : " + peopleArr[idx]);
        sc.close();
    }
}
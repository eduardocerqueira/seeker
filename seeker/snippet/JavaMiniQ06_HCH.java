//date: 2023-02-07T16:52:09Z
//url: https://api.github.com/gists/e06a04084374a628bd94f9c0d004f9f6
//owner: https://api.github.com/users/Chan-hee822

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Random;

public class Pr6 {
    public static void main(String[] args){

        /*
        HashMap map = new HashMap<>();
        map.put(1,"이재명");
        map.put(2,"윤셕열");
        map.put(3,"심삼정");
        map.put(4,"안철수");
        * */

        String [] candy = {"무효","이재명","윤석열","심삼정","안철수"};
        int [] score = {0,0,0,0,0};
        int cnt = 1;
        int ranNum = 0;
        int lastOne = 0;
        int lastCnt = 0;
        while(cnt <=10000){
            ranNum = new Random().nextInt((4)) + 1;
            if(ranNum == 1){
                score[ranNum] += 1;
            }else if(ranNum == 2){
                score[ranNum] += 1;
            } else if (ranNum == 3) {
                score[ranNum] += 1;
            }else if(ranNum==4) {
                score[ranNum] += 1;
            }
            lastOne = ranNum;
            lastCnt = cnt;
            int checkDouble = 0;
            int maxScore2 = Arrays.stream(score).max().getAsInt();
            
            //다중 최다 득표 제거
            if(lastCnt==10000){
                for (int i = 1; i < candy.length; i++) {
                    if(score[i] == maxScore2){
                        checkDouble +=1;
                    }
                }
                if(checkDouble > 1) {
                    cnt--;
                    score[lastOne] -=1;
                }
            }

            cnt++;
        }
        double vRatio = ((double)(lastCnt)/10000) * 100;

        System.out.println("{투표 시뮬레이션}");
        System.out.printf("[투표진행율]: %05.2f, %d 투표=> %s",(vRatio),(lastCnt),(candy[lastOne]));
        System.out.println();
        for (int i = 1; i < candy.length; i++) {
            System.out.printf("[기호%d] %s: %05.2f (투표수: %d)\n",(i),(candy[i]),((double)score[i]/10000)*100,(score[i]));
        }

        int maxScore = Arrays.stream(score).max().getAsInt();
        int maxVotePerson = 0;
        for (int i = 1; i < candy.length; i++) {
            if(score[i] == maxScore){
                maxVotePerson = i;
            }
        }
        if(lastCnt == 10000){
            System.out.printf("[투표결과]당선인 : %s",(candy[maxVotePerson]));
        }
    }
}

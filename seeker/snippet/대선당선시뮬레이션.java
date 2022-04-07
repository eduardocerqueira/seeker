//date: 2022-04-07T16:50:16Z
//url: https://api.github.com/gists/e5e3420a62e60437e596b25d1a957ea9
//owner: https://api.github.com/users/AgFe2

import java.util.*;

public class 대선당선시뮬레이션 {
    public static void solution() {
        String[] candidate = {"이재명", "윤석열", "심상정", "안철수"};
        HashMap<Integer, Integer> getVoted = new HashMap<>();
        Random vote = new Random();
        int peopleNum = 10000;
        int electedNum = -1;
        String electedPer = "";

        //후보자 번호별 득표수 기록
        getVoted.put(1, 0);
        getVoted.put(2, 0);
        getVoted.put(3, 0);
        getVoted.put(4, 0);


        //랜덤으로 투표, 득표 기록, 진행율 출력
        for (int i = 0; i < peopleNum; i++) {
            int num = vote.nextInt(4);
            double progress = ((double)(i + 1) / peopleNum) * 100;
            System.out.printf("[투표진행율]: %05.2f%%, %d명 투표 => %s", progress, i+1, candidate[num]);
            System.out.println();
            switch (num){
                case 0:
                    getVoted.replace(1, getVoted.get(1) + 1);
                    break;
                case 1:
                    getVoted.replace(2, getVoted.get(2) + 1);
                    break;
                case 2:
                    getVoted.replace(3, getVoted.get(3) + 1);
                    break;
                case 3:
                    getVoted.replace(4, getVoted.get(4) + 1);
                    break;
            }
            for (int j = 1; j < 5; j++) {
                System.out.printf("[기호:%d] %s: %05.2f%%, (투표수: %d)", j, candidate[j-1], getVoted.get(j)*100.0 / (i+1), getVoted.get(j));
                System.out.println();
            }
        }

        //동률인 경우에는 당선자가 없는 것으로 처리
        int SameVote[] = {0, 0, 0, 0};
        //Hashmap의 value값(득표수)만 Array로 변경
        for (int i = 1; i < 5; i++) {
            SameVote[i-1] = (getVoted.get(i));
        }
        //득표수 오름차순 정리
        Arrays.sort(SameVote);
        //동률 비교
        if (SameVote[2] == SameVote[3]) {
            electedPer = "동률이므로 당선자는 없습니다.";
        } else {
            for (int i: getVoted.keySet()) {
                if (getVoted.get(i) == SameVote[3]) {
                    electedNum = i;
                    electedPer = candidate[electedNum - 1];
                }
            }
        }

        System.out.println("[투표결과] 당선인: " + electedPer);


    }

    public static void main(String[] arg) {
        solution();
    }
}
//date: 2022-09-07T17:21:02Z
//url: https://api.github.com/gists/c464c0c84a5d03d1ba7d8db62af70421
//owner: https://api.github.com/users/junga970

// 

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

public class JavaStudy06 {
    public static void main(String[] args) {
        Random rnd = new Random();

        String[] candidates = new String[]{"이재명", "윤석열", "심상정", "안철수"};
        Integer[] cnts = new Integer[]{0, 0, 0, 0};

        int total = 0;
        for (int i = 0; i < 10000; i++) {
            // 랜덤 투표
            int voting = rnd.nextInt(4);
            cnts[voting]++;
            total++;

            // 투표진행율
            System.out.printf("\n[투표진행율]: %.2f%%, %d명 투표 => %s\n", (double)total / 100, total, candidates[voting]);

            // 후보별 진행율
            System.out.printf("[기호:1] 이재명: %.2f%%, (투표수: %d)\n", (double)cnts[0] / 100, cnts[0]);
            System.out.printf("[기호:2] 윤석열: %.2f%%, (투표수: %d)\n", (double)cnts[1] / 100, cnts[1]);
            System.out.printf("[기호:3] 심상정: %.2f%%, (투표수: %d)\n", (double)cnts[2] / 100, cnts[2]);
            System.out.printf("[기호:4] 안철수: %.2f%%, (투표수: %d)\n", (double)cnts[3] / 100, cnts[3]);
        }
        
        // 동률 확인
        Integer[] cntsSort = cnts.clone();
        Arrays.sort(cntsSort);

        ArrayList topCandidates = new ArrayList();
        for (int i = 0; i < cntsSort.length; i++) {
            if (cnts[i] == cntsSort[3]) {
                topCandidates.add(candidates[i]);
            }
        }

        // 투표 결과
        if (topCandidates.size() > 1) {
            System.out.printf("%s 후보가 동률입니다.", topCandidates);
        }
        else {
            System.out.printf("[투표결과] 당선인: %s", topCandidates.get(0));
        }
    }
}
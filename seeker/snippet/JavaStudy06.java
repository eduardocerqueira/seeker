//date: 2023-06-09T17:04:22Z
//url: https://api.github.com/gists/b6d57461e6a684c11e287bc5fb19cd3b
//owner: https://api.github.com/users/ssun2kim

/*
    김태양
 */
import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;

public class JavaStudy06 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("총 진행할 투표수를 입력해 주세요.");
        int voteCnt = sc.nextInt();
        System.out.print("가상 선거를 진행할 후보자 인원을 입력해 주세요.");
        int candiCnt = sc.nextInt();

        ArrayList candilist = new ArrayList();
        for (int i = 0; i < candiCnt; i++) {
            System.out.print((i + 1) + "번째 후보자이름을 입력해 주세요.");
            candilist.add(sc.next());
        }
        System.out.println();
        int cnt = 1;
        Random random = new Random();
        ArrayList<Integer> voteCntEach = new ArrayList<>();
        for (int i = 0; i < candiCnt; i++) {
            voteCntEach.add(i, 0);
        }

        while (cnt <= voteCnt) {
            double voteRate = (double) cnt / voteCnt * 100;
            String candiRandom = candilist.get(random.nextInt(candiCnt)).toString();
            System.out.printf("[투표진행률]: %.2f%%, %d명 투표 => %s\n",
                    voteRate, cnt, candiRandom);

            for (int i = 0; i < candiCnt; i++) {
                if (candilist.get(i).equals(candiRandom)) {
                    voteCntEach.set(i, voteCntEach.get(i) + 1);
                }
            }

            for (int i = 0; i < candiCnt; i++) {
                double voteRateEach = (double) voteCntEach.get(i) / voteCnt * 100;
                System.out.printf("[기호:" + (i + 1) + "] %s:\t %.2f%%\t (투표수: %d)\n",
                        candilist.get(i), voteRateEach, voteCntEach.get(i));
            }
            cnt++;
            System.out.println();
        }
        int resultIdx = voteCntEach.indexOf(voteCntEach
                .stream().max((a, b) -> a - b).get());
        String result = candilist.get(resultIdx).toString();
        System.out.printf("[투표결과] 당선인 : %s\n", result);
    }
}
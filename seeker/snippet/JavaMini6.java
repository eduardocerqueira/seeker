//date: 2024-02-06T17:02:10Z
//url: https://api.github.com/gists/aa67c22eb46e19e019fb8e1c09933845
//owner: https://api.github.com/users/p-yo00

/**
 *  박예온
 *  미니과제 6: 선거 시뮬레이션
 */
import java.util.Random;
import java.util.Scanner;

public class JavaMini6 {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        System.out.print("총 진행할 투표수를 입력해 주세요.");
        int voteCnt = scanner.nextInt();
        System.out.print("가상 선거를 진행할 후보자 인원을 입력해 주세요.");
        int candiCnt = scanner.nextInt();
        scanner.nextLine();

        String[] candiArr = new String[candiCnt];
        int[] candiScore = new int[candiCnt];

        for (int i=0; i<candiCnt; i++) {
            System.out.printf("%d번째 후보자이름을 입력해 주세요.", i+1);
            candiArr[i] = scanner.nextLine();
        }
        System.out.println();

        Random random = new Random();
        for (int i=0; i<voteCnt; i++) {
            int vote = random.nextInt(candiCnt);
            candiScore[vote]++;
            System.out.printf("[투표진행률]: %.2f%%, %d명 투표 => %s\n", (float)(i+1)/voteCnt*100, i+1, candiArr[vote]);
            for (int j=0; j<candiCnt; j++) {
                System.out.printf("[기호:%d]\t%s:\t%.2f%%\t(투표수: %d)\n", j+1, candiArr[j], (float)candiScore[j]/voteCnt*100, candiScore[j]);
            }
            System.out.println();
        }
        int maxIdx = 0;
        for (int i=1; i<candiCnt; i++) {
            if (candiScore[i] > candiScore[maxIdx]) {
                maxIdx = i;
            }
        }
        System.out.printf("[투표결과] 당선인 : %s", candiArr[maxIdx]);
    }
}
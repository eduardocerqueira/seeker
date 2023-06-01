//date: 2023-06-01T16:47:59Z
//url: https://api.github.com/gists/ca75748db87baa198e60be0783e317ca
//owner: https://api.github.com/users/ParkKangRak

/*
박강락
 */
import java.util.Random;
import java.util.Scanner;

public class JavaStudy06 {

    public static void main(String[] args) {
        Scanner scan = new Scanner(System.in);
        Random random = new Random();
        int sum = 0;
        int selectPeople = 0;

        System.out.print("총 진행할 투표수를 입력해 주세요: ");
        int sumVote = scan.nextInt();

        System.out.print("가상 선거를 진행할 후보자 인원을 입력해 주세요: ");
        int numPeople = scan.nextInt();

        String[] namePeople = new String[numPeople];
        int[] countPeople = new int[numPeople];

        for (int i = 0; i < namePeople.length; i++) {
            System.out.printf("%d번째 후보자 이름을 입력해 주세요: ", i + 1);
            namePeople[i] = scan.next();
        }

        boolean hasDuplicateMax = false;

        do {
            // 초기화
            sum = 0;
            countPeople = new int[numPeople];
            hasDuplicateMax = false;

            for (int i = 1; i <= sumVote; i++) {
                int num = random.nextInt(numPeople);
                countPeople[num]++;
                sum++;
                System.out.printf("[투표진행률]: %.2f%%, %d명 투표 => %s\n", (double) sum / sumVote * 100, i, namePeople[num]);
                for (int j = 0; j < namePeople.length; j++) {
                    System.out.printf("[기호:%d] %2s:  %5.2f%%  (투표수:%d)\n", j + 1, namePeople[j],
                            (double) countPeople[j] / sumVote * 100, countPeople[j]);
                }
                System.out.println();
            }

            int maxVotes = 0;
            int maxCandidateCount = 0;

            for (int j = 0; j < countPeople.length; j++) {
                if (countPeople[j] > maxVotes) {
                    maxVotes = countPeople[j];
                    maxCandidateCount = 1;
                } else if (countPeople[j] == maxVotes) {
                    maxCandidateCount++;
                }
            }

            if (maxCandidateCount > 1) {
                hasDuplicateMax = true;
                System.out.println("최대 득표자가 동일하여 재투표합니다..\n");
            }

        } while (hasDuplicateMax);

        int maxVotes = 0;
        for (int i = 0; i < countPeople.length; i++) {
            if (countPeople[i] > maxVotes) {
                maxVotes = countPeople[i];
                selectPeople = i;
            }
        }

        System.out.printf("[투표결과] 당선인: %s\n", namePeople[selectPeople]);
    }
}

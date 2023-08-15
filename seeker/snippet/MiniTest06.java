//date: 2023-08-15T16:59:00Z
//url: https://api.github.com/gists/390ba9284afa9063b152dbf060eb74a5
//owner: https://api.github.com/users/Seungmi97

/*
황승미
제로베이스 백엔드 스쿨 16기
*/

import java.io.IOException;
import java.util.*;

class Candidate {
    String name;
    int vote;

    Candidate(String name, int vote) {
        this.name = name;
        this.vote = vote;
    }
}

public class Main {
    public static void main(String[] args) throws IOException {
        Scanner sc = new Scanner(System.in);
        Random random = new Random();
        Candidate winner = new Candidate("", 0);

        try {
            System.out.println("[가상 선거 및 당선 시뮬레이션 프로그램]");
            System.out.print("총 진행할 투표수를 입력해 주세요.");
            int total = sc.nextInt();
            System.out.print("가상 선거를 진행할 후보자 인원을 입력해 주세요.");
            int num = sc.nextInt();
            Candidate[] candidate = new Candidate[num];
            for (int i = 0; i < num; i++) {
                System.out.printf("%d번째 후보자 이름을 입력해 주세요.", i + 1);
                candidate[i] = new Candidate(sc.next(), 0);
            }
            System.out.println();

            for (int i = 0; i < total; i++) {
                int pick = random.nextInt(num);
                candidate[pick].vote++;
                System.out.printf("[투표진행률]: %.2f%%, %d명 투표 => %s\n", (float)(i + 1) * 100 / total, i + 1, candidate[pick].name);
                for (int j = 0; j < num; j++) {
                    System.out.printf("[기호:%d] %s:\t%.2f%%\t(투표수: %d)\n", j + 1, candidate[j].name, (float)candidate[j].vote * 100 / total, candidate[j].vote);
                    winner = candidate[j].vote > winner.vote ? candidate[j] : winner;
                }
                System.out.println();
            }

            System.out.print("[투표결과] 당선인 : ");
            System.out.println(winner.name);
        } catch(Exception e) {
            System.out.println("e = " + e);
        }
    }
}
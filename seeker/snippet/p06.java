//date: 2024-03-14T16:58:57Z
//url: https://api.github.com/gists/f939e6fce4d10b51ad2dacc04a8ac755
//owner: https://api.github.com/users/jjm159

package org.example.assignment_03_18;

/* 정재명
*
* 가상 선거 및 당선 시뮬레이션 프로그램
*
* 1. 총 투표를 진행할 투표수를 입력 받음
* 2. 선거를 진행할 후보자 수를 입력 받고, 이에 대한 이름을 입력 받음
* 3. 각 입력받은 후보자는 순서대로 기호1, 기호2, 기호3… 형식으로 기호번호 부여함
* 4. 각 투표수의 결과는 선거를 진행할 후보자를 동일한 비율로 랜덤하게 발생
* 5. 임의번호는 Random함수의 nextInt()함수를 통해서 생성
* 6. 1표에 대한 투표한 결과에 대해서 투표자와 이에 대한 결과를 화면 출력해야 함
*
* 아래 내용은 전제조건으로 진행
* - 투표수는 1 ~ 10000 사이의 값을 입력하며, 그외 값 입력에 대한 예외는 없다고 가정함.
* - 후보자 인원은 2 ~ 10 사이의 값을 입력받으면, 그외 값 입력에 대한 예외는 없다고 가정함.
* - 후보자이름은 한글로 입력하며, 10자 미만으로 입력함. (역시, 그외 입력에 대한 예외는 없다고
* 가정함.)
*
* */

import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.Scanner;

public class p06 {
    public static void main(String[] args) {
        Vote vote = new Vote();
        vote.inputTotalCount();
        vote.inputCadidateOption();
        vote.printResult();
    }
}

class Vote {

    private final HashMap<Integer, String> candidateNameTable = new HashMap<>();
    private final HashMap<String, Integer> candidateVoteCountTable = new HashMap<>();
    private int totalCandidateNumber = 0;
    private final Scanner sc = new Scanner(System.in);
    private final Random rd = new Random();
    private int totalVoteCount = 0;

    public void inputTotalCount() {
        System.out.print("총 진행할 투표수를 입력해 주세요.");
        this.totalVoteCount = sc.nextInt();
        sc.nextLine();
    }

    public void inputCadidateOption() {
        System.out.print("가상 선거를 진행할 후보자 인원을 입력해 주세요.");
        int count = sc.nextInt();
        sc.nextLine();
        this.totalCandidateNumber = count;

        for (int i = 0; i < count; i++) {
            String content = String.format("%d번째 후보자이름을 입력해 주세요.", i + 1);
            System.out.print(content);
            String candidateName = sc.nextLine();
            candidateVoteCountTable.put(candidateName, 0);
            candidateNameTable.put(i, candidateName);
        }
    }

    public void printResult() {
        for (int i = 0; i < totalVoteCount; i++) {
            int index = cast();
            String name = candidateNameTable.get(index);
            int currentCount = candidateVoteCountTable.get(name);
            candidateVoteCountTable.put(name, currentCount + 1);
            printProcess(name, i + 1);
        }
        printFianlElectedCandidate();
    }

    private void printProcess(
            String currentCastedName,
            int currentCastedTotalCount
    ) {
        double progressPercentage = (double) currentCastedTotalCount / totalVoteCount * 100;
        String currentTitle = String.format(
                "[투표진행률]: %.2f%%, %d명 투표 => %s",
                progressPercentage,
                currentCastedTotalCount,
                currentCastedName
        );
        System.out.println(currentTitle);

        for (int i = 0; i < totalCandidateNumber; i++) {
            String name = candidateNameTable.get(i);
            int count = candidateVoteCountTable.get(name);
            double percentage = (double) count / totalVoteCount * 100;
            String result = String.format(
                    "[기호: %d] %s\t%.2f%%\t(투표수: %d)",
                    i + 1,
                    name,
                    percentage,
                    count
            );
            System.out.println(result);
        }

        System.out.println();
    }

    private int cast() {
        return rd.nextInt(0, totalCandidateNumber);
    }

    public void printFianlElectedCandidate() {
        String electedCandidate = getElectedCandidate();
        String result = String.format("[투표결과] 당선인: %s", electedCandidate);
        System.out.println(result);
    }

    private String getElectedCandidate() {
        return candidateVoteCountTable.entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .map(Map.Entry::getKey)
                .orElse("");
    }

}


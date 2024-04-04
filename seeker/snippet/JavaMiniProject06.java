//date: 2024-04-04T17:04:19Z
//url: https://api.github.com/gists/93d54c05a45855034b80635a9db30f8c
//owner: https://api.github.com/users/dlrkdkhs

/*
  이가온
  미니 과제 6
*/

import java.util.*;


class ElectionData {
    String name; // 후보자 이름
    int intVotes; // 투표수
    double douVotes; // 개별 투표율
    ElectionData(String name, int intVotes, double douVotes){
        this.name = name;
        this.intVotes = intVotes;
        this.douVotes = douVotes;
    }

}

public class MiniProject6 {
    public static void main(String[] args) {
        Scanner scan = new Scanner(System.in);

        System.out.print("총 진행할 투표수를 입력해 주세요.");
        int personnel = scan.nextInt();

        System.out.print("가상 선거를 진행할 후보자 인원을 입력해주세요.");
        int candidate = scan.nextInt();
        scan.nextLine();
        double vote = (double)100/personnel;
        ElectionData[] datas = getElectionData(candidate, scan);
        startVote(datas, personnel, candidate, vote);
        elected(datas);

    }

    // 후보자 설정
    public static ElectionData[] getElectionData(int candidate, Scanner scan){
        ElectionData[] datas = new ElectionData[candidate];
        for(int i = 0; i < candidate; i++){
            System.out.print(i + 1 + "번째 후보자이름을 입력해 주세요.");
            String name = scan.nextLine();
            ElectionData data = new ElectionData(name, 0,0.0);
            datas[i] = data;
        }
        return datas;

    }

    // 투표진행
    public static void startVote(ElectionData[] datas, int personnel, int candidate, double vote){
        Random random = new Random();
        double totalVote = 0.0;
        for(int i = 0; i < personnel; i++){
            int choice = random.nextInt(candidate);
            totalVote += vote;
            System.out.printf("[투표진행률]: %.2f%%, %d명 투표 => %s\n", totalVote, i + 1, datas[choice].name);
            datas[choice].douVotes += vote;
            datas[choice].intVotes++;
            printInfo(datas);
            System.out.println();
        }
    }

    // 투표 중간 정보
    public static void printInfo(ElectionData[] datas){
        int i = 1;
        for(ElectionData data : datas){
            System.out.printf("[기호:%d] %s:\t%.2f%%\t(투표수: %d)\n",i, data.name, data.douVotes, data.intVotes);
            i++;
        }
    }

    // 투표결과
    public static void elected(ElectionData[] datas){
        int max = 0;
        for(int i = 1; i < datas.length; i++){
            if(datas[i].intVotes > datas[max].intVotes){
                max = i;
            }
        }
        System.out.println("[투표결과] 당선인 : " + datas[max].name);
    }
}

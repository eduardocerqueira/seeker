//date: 2024-04-24T16:49:07Z
//url: https://api.github.com/gists/af77cd3d152f59d7527b1e533e7126ac
//owner: https://api.github.com/users/GahyungKim

import java.util.*;

public class Q2 {

    public static HashMap voting(int votNum, List candidName){
        HashMap candidIdx = new HashMap<>();
        for (int i = 1; i < candidName.size()+1; i++) {
            candidIdx.put(i,candidName.get(i-1));
        }
        return candidIdx;
    }

    public static HashMap simulation(int votNum, HashMap candidIdx){
        Random random = new Random();
        HashMap<Integer, Integer> votCntMap = new HashMap<Integer, Integer>();
        Double currentVot;
        Double votNumD = new Double(votNum);
        for (int i = 1; i < candidIdx.size()+1; i++) {
            votCntMap.put(i, 0);
        }

        for (int i = 1; i < votNum+1; i++) {
            int randVot = random.nextInt(candidIdx.size())+1;
            votCntMap.put(randVot , votCntMap.get(randVot)+1);

            currentVot = (i/votNumD)*100;

            System.out.println("[투표진행률]: "+String.format("%.2f", currentVot)+"%, "+i+"명 투표 ==> "+candidIdx.get(randVot));

            for(int key:votCntMap.keySet()){
                Double votper = (votCntMap.get(key)/votNumD)*100;
                String votperStr = String.format("%.2f", votper)+"%";
                votperStr = String.format("%-6s", votperStr);
                System.out.println("[기호:"+key+"] "+String.format("%-6s",candidIdx.get(key)+":")+ votperStr+" (투표수: "+votCntMap.get(key)+")");
            }
            System.out.println();
        } return votCntMap;
    }

    public static void main(String[] args){
        Scanner sc = new Scanner(System.in);
        System.out.print("총 진행할 투표수를 입력해 주세요.");
        int votNum = sc.nextInt();
        System.out.print("가상 선거를 진행할 후보자 인원을 입력해 주세요.");
        int candidNum = sc.nextInt();

        List candidName = new ArrayList();
        for (int i = 1; i < candidNum+1; i++) {
            System.out.print(i+"번째 후보자이름을 입력해 주세요.");
            candidName.add(sc.next());
        }
        HashMap<Integer, String> candidIdx = voting(votNum, candidName);
        System.out.println();
        HashMap<Integer,Integer> votCntMap = simulation(votNum, candidIdx);

        String winner = null;
        Integer maxNum = 0;
        for(int key: votCntMap.keySet()){
            if (votCntMap.get(key) > maxNum){
                maxNum = votCntMap.get(key);
                winner = candidIdx.get(key);
            }
        }
        System.out.println("[투표결과] 당선인 : "+ winner);

    }
}

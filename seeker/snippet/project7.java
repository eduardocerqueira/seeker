//date: 2023-09-08T16:43:54Z
//url: https://api.github.com/gists/8ca9087566d26e9e8d43e5dbbf53fb35
//owner: https://api.github.com/users/cjwon0827

import java.util.*;

public class project7 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        HashSet<Integer> numSet = new HashSet<>();
        ArrayList<Integer> numSetList;

        while (numSet.size() != 6){
            numSet.add((int) (Math.random() * 45 + 1));
        }
        numSetList = new ArrayList<>(numSet);
        numSetList.sort(Comparator.naturalOrder());

        HashMap<Character, ArrayList<Integer>> lottoMap = new HashMap<>();
        System.out.println("[로또 당첨 프로그램]\n");
        System.out.print("로또 개수를 입력해 주세요.(숫자 1 ~ 10):");
        int lottoNum = sc.nextInt();

        for(char i = 'A'; i < 'A' + lottoNum; i++){
            HashSet<Integer> set = new HashSet<>();
            while (set.size() != 6){
                set.add((int) (Math.random() * 45 + 1));
            }
            ArrayList<Integer> setList = new ArrayList<>(set);
            setList.sort(Comparator.naturalOrder());
            System.out.println(i + "\t" + setList);
            lottoMap.put(i, setList);
        }
        System.out.println();
        System.out.println("[로또 발표]");
        System.out.println("\t" + numSetList + "\n");

        for(char i = 'A'; i < 'A' + lottoNum; i++){
            int count = 0;
            ArrayList<Integer> list = lottoMap.get(i);

            for(int j = 0; j < 6; j++){
                for(int k = 0; k < 6; k++){
                    if(numSetList.get(j) == list.get(k)){
                        count++;
                    }
                }
            }
            System.out.println(i + "\t" + list + " => " + count + "개 일치");
        }
        sc.close();
    }
}
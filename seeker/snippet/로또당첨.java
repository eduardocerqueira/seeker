//date: 2022-04-07T16:50:45Z
//url: https://api.github.com/gists/186e59e9b0e24ea96a473a067a583bc0
//owner: https://api.github.com/users/AgFe2

import java.util.*;

public class 로또당첨 {
    public static void solution() {
        int rep = 0;

        HashMap<Character, ArrayList> myLotto = new HashMap<>();
        HashMap<Character, Integer> myLottoSame = new HashMap<>();

        System.out.println("[로또 당첨 프로그램]");
        System.out.println();


        //로또 개수 입력
        while (rep > 10 || rep <1) {
            try {
                System.out.print("로또 개수를 입력해 주세요.(숫자 1 ~ 10):");
                Scanner sc = new Scanner(System.in);
                rep = sc.nextInt();
                if (rep >= 1 && rep <= 10) {
                    break;
                }
                System.out.println("1~10사이의 숫자만 입력해주세요.");
            } catch (InputMismatchException e) {
                System.out.println("1~10사이의 숫자만 입력해주세요.");
            }
        }

        //로또 생성
        for (int i = 0; i < rep; i++) {
            myLotto.put((char)((int)'A' + i), lotto());
        }
        for (int i = 0; i < rep; i++) {
            System.out.print((char)((int)'A' + i) + "  ");
            for (int j = 0; j < 6; j++) {
                System.out.printf("%02d", myLotto.get((char)((int)'A' + i)).get(j));
                if (j == 5) {
                    System.out.println();
                    break;
                } else {
                    System.out.print(",");
                }
            }
        }
        System.out.println();

        //오늘의 로또 번호
        System.out.println("[로또 발표]");
        System.out.print("   ");
        ArrayList todayLotto = new ArrayList(lotto());
        for (int i = 0; i < 6; i++) {
            System.out.printf("%02d", todayLotto.get(i));
            if (i == 5) {
                System.out.println();
                break;
            } else {
                System.out.print(",");
            }
        }
        System.out.println();

        //로또 확인
        System.out.println("[내 로또 결과]");
        for (int i = 0; i < rep; i++) {
            int sameCnt = 0;
            for (int j = 0; j < todayLotto.size(); j++) {
                for (int k = 0; k < 6; k++) {
                    if((int)todayLotto.get(j) == (int)myLotto.get((char)((int)'A' + i)).get(k)) {
                        sameCnt++;
                    }
                }
            }
            myLottoSame.put((char)((int)'A' + i), sameCnt);
        }

        //결과 출력
        for (int i = 0; i < rep; i++) {
            System.out.print((char)((int)'A' + i) + "  ");
            for (int j = 0; j < 6; j++) {
                System.out.printf("%02d", myLotto.get((char)((int)'A' + i)).get(j));
                if (j == 5) {
                    break;
                } else {
                    System.out.print(",");
                }
            }
            System.out.print(" => " + myLottoSame.get((char)((int)'A' + i)) + "개 일치");
            System.out.println();
        }
        System.out.println();

    }


    public static ArrayList lotto() {
        HashSet set = new HashSet();

        for (int i = 0; set.size() < 6; i++) {
            int num = (int)(Math.random() * 45) + 1;
            set.add(num);
        }

        ArrayList list = new ArrayList(set);
        Collections.sort(list);

        return list;
    }

    public static void main(String[] arg) {
        solution();
    }
}
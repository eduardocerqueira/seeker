//date: 2025-01-10T17:00:40Z
//url: https://api.github.com/gists/ab3e7a9c6db719a92e9d0af3f429b5de
//owner: https://api.github.com/users/YB-Taekwon

import java.util.*;

public class Lotto {
    // 로또 번호 생성
    private static List<Set<Integer>> generateLottoNumber(int n) {
        List<Set<Integer>> lottoNumbers = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            Set<Integer> lotto = new TreeSet<>();
            while (lotto.size() < 6) {
                int lottoNumber = (int) (Math.random() * 45) + 1;
                lotto.add(lottoNumber);
            }
            lottoNumbers.add(lotto);
        }
        return lottoNumbers;
    }


    // 출력 시, 대괄호 제거
    private static String formatLottoNumbers(Set<Integer> lotto) {
        return lotto.toString().substring(1, lotto.toString().length() - 1); // 대괄호 제거
    }


    public static void main(String[] args) {
        System.out.println("[로또 당첨 프로그램]");
        System.out.println();

        Scanner sc = new Scanner(System.in);
        System.out.print("로또 개수를 입력해주세요. (숫자 1 ~ 10): ");
        int n = sc.nextInt();
        while (n < 1 || n > 10) {
            System.out.println("최소 구매 수량은 1매이며, 최대 구매 수량은 10매입니다. 확인 후 다시 입력해주세요.");
            System.out.print("로또 개수를 입력해주세요. (숫자 1 ~ 10): ");
            n = sc.nextInt();
        }


        // 로또 리스트 및 로또 번호 생성 (메서드 호출)
        List<Set<Integer>> myLotto = generateLottoNumber(n);
        for (int i = 0; i < myLotto.size(); i++) {
            System.out.println((char) (65 + i) + "\t" + formatLottoNumbers(myLotto.get(i)));
        }
        System.out.println();


        // 당첨 로또 번호 생성 (기존 로또 생성 메서드 호출)
        Set<Integer> winningNumber = generateLottoNumber(1).get(0);
        System.out.println("[로또 발표]");
        System.out.println("\t" + formatLottoNumbers(winningNumber));
        System.out.println();


        // 로또 결과 비교
        System.out.println("[내 로또 결과]");
        for (int i = 0; i < myLotto.size(); i++) {
            // 복사본 생성 후 교집합으로 일치하는 항목 비교
            Set<Integer> match = new HashSet<>(myLotto.get(i));
            match.retainAll(winningNumber);
            int matchCount = match.size();
            System.out.println((char) (65 + i) + "\t" + formatLottoNumbers(myLotto.get(i)) + " => " + matchCount + "개 일치");
        }

    }
}
//date: 2024-03-14T16:59:21Z
//url: https://api.github.com/gists/5c73988c0f42916aef327acd43c60e75
//owner: https://api.github.com/users/jjm159

package org.example.assignment_03_18;

/* 정재명
*
* 로또 당첨 프로그램
*
* 1. 로또 구매 수량 입력
* 2. 입력한 개수만큼의 로또 개수 생성
* 3. 로또 당첨 번호 생성(숫자값은 중복 배제 및 정렬해서 표시)
* 4. 당첨 번호와 구매 로또 비교하여 숫자 일치 여부 판단
* 5. Collections.shuffle 함수 사용 금지!(shuffle함수는 과제의 취지와 맞지 않기 때문에, 사용시 0
*
* * */

import java.util.*;
import java.util.stream.IntStream;

public class p07 {
    public static void main(String[] args) {
        Lotto lotto = new Lotto();
        lotto.inputLottoNumberToBuy();
        lotto.printPurchasedLotto();
        lotto.printLottoResult();
        lotto.printMyResult();
    }
}

class Lotto {

    private final Scanner sc = new Scanner(System.in);
    private final Random rd = new Random();
    private final String[] alphabetIndices = { "A", "B", "C", "D", "E", "F", "G", "H", "I", "J" };
    private int buyCount = 0;

    private List<List<Integer>> purchasedLottoList;
    private List<Integer> winningLottoNumberList;


    public void inputLottoNumberToBuy() {
        System.out.println("[로또 당첨 프로그램]");
        System.out.print("로또 개수를 입력해 주세요.(숫자 1 ~ 10):");
        this.buyCount = sc.nextInt();
        sc.nextLine();
    }

    public void printPurchasedLotto() {
        this.purchasedLottoList = IntStream.range(0, buyCount)
                .mapToObj(i -> getLottoResult())
                .toList();
        for (int i = 0; i< purchasedLottoList.size(); i++) {
            List<Integer> lottoNumberList = purchasedLottoList.get(i);
            String lottoString = getLottoNumberString(lottoNumberList);
            String lineString = String.format(
                    "%s\t%s",
                    alphabetIndices[i],
                    lottoString
            );
            System.out.println(lineString);
        }
        System.out.println();
    }

    public void printLottoResult() {
        System.out.println("[로또 발표]");
        this.winningLottoNumberList = getLottoResult();
        String lottoString = getLottoNumberString(this.winningLottoNumberList);
        String result = String.format("\t%s", lottoString);
        System.out.println(result);
        System.out.println();
    }

    public void printMyResult() {
        System.out.println("[내 로또 결과]");
        for (int i = 0; i< purchasedLottoList.size(); i++) {
            List<Integer> lottoNumberList = purchasedLottoList.get(i);
            String lottoString = getLottoNumberString(lottoNumberList);
            int sameCount = getSameCount(lottoNumberList);
            String lineString = String.format(
                    "%s\t%s => %d개 일치",
                    alphabetIndices[i],
                    lottoString,
                    sameCount
            );
            System.out.println(lineString);
        }
        System.out.println();
    }

    private List<Integer> getLottoResult() {
        Set<Integer> result = new HashSet<>();
        while (result.size() < 6) {
            result.add(getLottoNumber());
        }
        return result.stream().sorted().toList();
    }


    private Integer getLottoNumber() {
        int min = 1;
        int max = 45;
        return rd.nextInt(min, max + 1);
    }

    private String getLottoNumberString(List<Integer> lottoNumberList) {
        StringJoiner joiner = new StringJoiner(",");
        lottoNumberList.stream()
                .map(number -> String.format("%02d", number))
                .forEach(joiner::add);
        return joiner.toString();
    }

    private int getSameCount(List<Integer> myNumberList) {
        int count = 0;
        for (int myNumber: myNumberList) {
            if (this.winningLottoNumberList.contains(myNumber)) {
                count++;
            }
        }
        return count;
    }

}


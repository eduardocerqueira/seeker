//date: 2023-03-03T17:06:19Z
//url: https://api.github.com/gists/aefd6563e3103b3a39b1d27e1d46fb47
//owner: https://api.github.com/users/hseungho

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

/**
 * 황승호
 * 과제 7. 로또 당첨 프로그램
 * Stream API를 적극적으로 써 본 과제다.
 * 하지만 이에 여러 군데에서 중복코드가 보이기에
 * 최적화 및 리팩토링에 대한 고려를 해야 할 듯 싶다.
 */
public class Lotto {

    public static void main(String[] args) {
        Lotto lotto = new Lotto();
        lotto.start();
    }

    private static final int LOTTO_ELEMENT_COUNT_MAX = 6;
    private static final int LOTTO_ELEMENT_MAX = 45;
    private final Scanner sc;
    /** 생성할 로또 복권 총 개수 */
    private int numOfLottos;
    /** 생성된 로또 결과를 정렬 및 중복 제거하여 저장하는 Set */
    private SortedSet<Integer> result;
    /** 생성된 로또 번호를 정렬 및 중복 제거하여 저장하는 Set */
    private SortedSet<Integer> lotto;
    /** 생성된 로또 번호를 저장한 Set을 저장하는 Map */
    private Map<Character, SortedSet<Integer>> myLottos;

    public Lotto() {
        sc = new Scanner(System.in);

        System.out.println("[로또 당첨 프로그램]\n");
    }

    public void start() {

        // 1. 로또 개수 입력
        numOfLottos = inputNumOfLotto();
        // 2. 개수 만큼 로또 생성
        createLotto();
        // 3. 로또 발표
        pickResultOfLotto();
        // 4. 내 로또 결과 출력
        matchMineToResult();

    }

    private void createLotto() {
        Random random = new Random();
        myLottos = new HashMap<>();

        // 입력한 총 로또 개수 만큼 반복.
        for (int i = 0, c = 65; i < numOfLottos; i++, c++) {
            lotto = new TreeSet<>();

            // 로또가 모두 6개가 될 때까지 로또 생성 및 저장
            while(lotto.size() < LOTTO_ELEMENT_COUNT_MAX) {
                // 로또 1 ~ 45 생성. random의 nextInt 함수는 bound 파라미터를 배제하기 때문에, 45+1로 전달.
                int r = random.nextInt(1, LOTTO_ELEMENT_MAX+1);
                lotto.add(r);
            }

            // 내 로또 리스트에 char 키(start with 'A`)에 로또 리스트를 저장
            myLottos.put((char)c, lotto);
        }
        printMyLottos();
    }

    private void printMyLottos() {
        StringBuilder builder = new StringBuilder();
        StringBuilder valueBuilder = new StringBuilder();

        // 내 로또 리스트를 루프
        myLottos.forEach((key, value) -> {
            AtomicInteger indexHolder = new AtomicInteger(); // 단순히 리스트의 인덱스를 확인하기 위한 객체

            // 1. key 출력
            builder.append(key).append("\t")
                    .append(value.stream()
                            .map(v -> {
                                // 2. 로또 번호 출력.
                                valueBuilder.append(v < 10 ? "0"+v : v);
                                if (indexHolder.getAndIncrement() < value.size() - 1) {
                                    valueBuilder.append(", ");
                                }
                                String lotto = valueBuilder.toString();
                                valueBuilder.setLength(0);
                                return lotto;
                            })
                            .collect(Collectors.joining()))
                    .append("\n");
        });
        System.out.println(builder);
    }

    private void pickResultOfLotto() {
        System.out.println("[로또 발표]");

        Random random = new Random();
        result = new TreeSet<>();

        // 로또 결과 생성
        while(result.size() < LOTTO_ELEMENT_COUNT_MAX) {
            int r = random.nextInt(1, LOTTO_ELEMENT_MAX+1);
            result.add(r);
        }

        StringBuilder builder = new StringBuilder("  \t");
        int size = result.size();
        AtomicInteger indexHolder = new AtomicInteger();
        result.forEach(v -> {
            builder.append(v < 10 ? "0"+v : v);
            if (indexHolder.getAndIncrement() < size - 1) {
                builder.append(", ");
            } else {
                builder.append("\n");
            }
        });
        System.out.println(builder);
    }

    private void matchMineToResult() {
        System.out.println("[내 로또 결과]");

        StringBuilder builder = new StringBuilder();
        myLottos.forEach((key, value) -> {
            builder.append(key).append("\t");

            AtomicInteger indexHolder = new AtomicInteger();
            AtomicInteger count = new AtomicInteger();
            value.stream()
                    .peek(v -> {
                        builder.append(v < 10 ? "0" + v : v);
                        if (indexHolder.getAndIncrement() < value.size() - 1) {
                            builder.append(", ");
                        }
                    })
                    .filter(v -> result.contains(v))
                    .forEach(v -> count.getAndIncrement());

            builder.append(" => ")
                    .append(count.get())
                    .append("개 일치")
                    .append("\n");
        } );
        System.out.println(builder);
    }

    private int inputNumOfLotto() {
        while (true) {
            int num = inputInteger("로또 개수를 입력해 주세요.(숫자 1 ~ 10):");
            if (num > 0 && num < 11) {
                return num;
            }
            System.out.println("ERROR: 로또 개수는 1부터 10 사이의 숫자만 입력해 주세요.");
        }
    }

    private int inputInteger(String question) {
        while (true) {
            try {
                System.out.print(question);
                return sc.nextInt();

            } catch (InputMismatchException e) {
                System.out.println("ERROR: 숫자로만 입력해주세요.");
                sc.nextLine();
            } catch (Exception e) {
                System.out.println("ERROR: 시스템에 오류가 발생하였습니다. 다시 시도해주세요.");
                System.exit(0);
            }
        }
    }


}

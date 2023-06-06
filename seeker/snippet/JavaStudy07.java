//date: 2023-06-06T17:08:56Z
//url: https://api.github.com/gists/6a0d45d164611d5730ee3e61109ad2ef
//owner: https://api.github.com/users/devkjy00


/*
 * 김주영
 * 로또 당첨 과제
 */
import java.util.*;
import java.util.stream.Collectors;


public class JavaStudy07{
	private static final int LOTTO_MAX_NUMBER = 45;
	private static final int ASC_A = 65;

	public static void main(String[] args) {
		Scanner sc = new Scanner(System.in);

		System.out.println("[로또 당첨 프로그램]\n");

		System.out.print("로또 개수를 입력해 주세요.(숫자 1 ~ 10):");
		int lottoCount = sc.nextInt();

		List<List<Integer>> lottos = generateLottos(lottoCount);
		printLottos(lottos);
		List<Integer> winningLotto = generateLotto();
		System.out.println("[로또 발표]");
		System.out.printf("\t\t%s\n\n", formatLotto(winningLotto));

		System.out.println("[내 로또 결과]");
		printResult(lottos, winningLotto);



	}

	private static void printResult(List<List<Integer>> lottos, List<Integer> winningLotto) {
		for (int i=0; i<lottos.size(); i++) {
			List<Integer> lotto = lottos.get(i);
			int matchCount = getMatchCount(lotto, winningLotto);
			System.out.printf("%c\t\t%s => %d개 일치 \n", (char)(ASC_A+i), formatLotto(lotto), matchCount);
		}
	}

	private static int getMatchCount(List<Integer> lotto, List<Integer> winningLotto) {
		int matchCount = 0;
		for (Integer integer : lotto) {
			if (winningLotto.contains(integer)) {
				matchCount++;
			}
		}
		return matchCount;
	}

	private static void printLottos(List<List<Integer>> lottos) {
		for (int i=0; i<lottos.size(); i++) {
			System.out.printf("%c\t\t%s\n", (char)(ASC_A+i), formatLotto(lottos.get(i)));
		}
		System.out.println();
	}

	private static String formatLotto(List<Integer> lotto) {
		return lotto.stream()
				.map(num -> String.format("%02d", num))
				.collect(Collectors.joining(","));
	}

	private static List<List<Integer>> generateLottos(int lottoCount) {
		List<List<Integer>> lottos = new ArrayList<>();

		for (int i = 0; i < lottoCount; i++) {
			List<Integer> lotto = generateLotto();
			lottos.add(lotto);
		}

		return lottos;
	}

	private static List<Integer> generateLotto(){
		Random random = new Random();
		Set<Integer> lotto = new HashSet<>();

		while (lotto.size() < 6) {
			int number = random.nextInt(LOTTO_MAX_NUMBER) + 1;
			lotto.add(number);
		}

		return lotto.stream().sorted().collect(Collectors.toList());
	}
}
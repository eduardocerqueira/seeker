//date: 2025-02-20T16:59:38Z
//url: https://api.github.com/gists/3e5654689efb2db025b3f1685c9e8284
//owner: https://api.github.com/users/fResult

import java.util.*;
import java.util.function.Predicate;
import java.util.stream.Stream;

class TestUtils {
  public static <Expected, Actual> void test(Expected expectedResult, Actual actualResult) {
    // System.out.printf("Expected=%s | Actual=%s\n", expectedResult, actualResult);
    final var expectedResultOpt = Optional.ofNullable(expectedResult);
    final var actualResultOpt = Optional.ofNullable(actualResult);

    if (!expectedResultOpt.equals(actualResultOpt)) {
      System.err.printf("Test failed: expected %s but got %s\n", expectedResult, actualResult);
    }
  }
}

/*  =============== BEGIN Related classes =============== */
public class MostAppearedDigitsProblem {
  public static void main(String... args) {
    final var actual1 = Problem.mostAppearedDigits(List.of(25, 2, 3, 57, 38, 41));
    final var expected1 = List.of(2, 3, 5);
    final var actual2 = Problem.mostAppearedDigits(List.of(250, 20, 30, 570, 38, 41));
    final var expected2 = List.of(0);
    TestUtils.test(expected1, actual1);
    TestUtils.test(expected2, actual2);
  }
}

class Problem {
  public static List<Integer> mostAppearedDigits(List<Integer> numbers) {
    final var numFreqWithMaximum =
        numbers.stream()
            .flatMap(Problem::toDigits)
            .reduce(new NumFreqWithMaximum(), Problem::toNumFreqWithMaximum, (x, y) -> x);
    final var highestFreq = Collections.max(numFreqWithMaximum.numberFreqMap().values());

    return numFreqWithMaximum.numberFreqMap().keySet().stream()
        .filter(Problem.isHighestFreqDigit(highestFreq, numFreqWithMaximum.numberFreqMap()))
        .toList();
  }

  private static NumFreqWithMaximum toNumFreqWithMaximum(
      NumFreqWithMaximum acc, Integer num) {

    /* NOTE: Use `int[]` instead of `int` to prevent error
     * `Variable 'maximum' is accessed from within inner class, needs to be final or effectively final`
     */
    final int[] maximum = {acc.maximum()};
    final var numberFreqMap =
        new HashMap<Integer, Integer>(acc.numberFreqMap()) {
          {
            final var frequencyToPut = getOrDefault(num, 1) + 1;
            maximum[0] = frequencyToPut > acc.maximum() ? frequencyToPut : acc.maximum();
            put(num, frequencyToPut);
          }
        };

    return new NumFreqWithMaximum(numberFreqMap, maximum[0]);
  }

  private static Stream<Integer> toDigits(Integer num) {
    return Arrays.stream(String.valueOf(num).split("")).map(Integer::valueOf);
  }

  private static Predicate<Integer> isHighestFreqDigit(int maximum, Map<Integer, Integer> hashMap) {
    return num -> Optional.ofNullable(hashMap.get(num)).equals(Optional.of(maximum));
  }
}

record NumFreqWithMaximum(Map<Integer, Integer> numberFreqMap, int maximum) {
  public NumFreqWithMaximum() {
    this(Map.of(), Integer.MIN_VALUE);
  }
}
/*  =============== END Related classes =============== */

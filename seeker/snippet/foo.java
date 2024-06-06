//date: 2024-06-06T16:57:30Z
//url: https://api.github.com/gists/56a6aa6e99c6a3175aed8fb64a3ef748
//owner: https://api.github.com/users/jeanbza

// This is nice and terse.
public static void main(String[] args){
  ArrayList<Pair<String, int>> test_scores = {{"Corey", 79}, {"Jason", 54}, {"Alice", 92}, {"Mark", 68}};
  test_scores.stream()
    .filter(s -> s.getValue() > 80)
    .forEach(s -> System.out.println(s.getKey() + " passed the test"));
}

// This is annoyingly verbose.
public static void main(String[] args){
  ArrayList<Pair<String, int>> test_scores = {{"Corey", 79}, {"Jason", 54}, {"Alice", 92}, {"Mark", 68}};
  test_scores.stream()
    .filter(passedTheTest)
    .forEach(printPassed);
}

public boolean passedTheTest(Pair<String, int> s) {
  return s.getValue() > 70
}

public boolean printPassed(Pair<String, int> s) {
  System.out.println(s.getKey() + " passed the test")
}
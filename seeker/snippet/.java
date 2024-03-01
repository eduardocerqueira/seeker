//date: 2024-03-01T17:07:27Z
//url: https://api.github.com/gists/0e1d5e5589e4c9b0626e340872d41853
//owner: https://api.github.com/users/jaimemin

Map<String, Integer> frequencyTable = new TreeMap<>();

for (String s : args) {
  frequencyTable.merge(s, 1, (count, incr) -> count + incr); // Lambda
}

System.out.println(frequencyTable);
//date: 2021-09-09T17:09:44Z
//url: https://api.github.com/gists/d021d2e95a7906459c0636f6e8a8736a
//owner: https://api.github.com/users/kunalgaurav18

List<Integer> numbers = Arrays.asList(new Integer[] {1,2,3,4,5});
numbers.stream().filter(num -> num%2 == 0).forEach(System.out::println);
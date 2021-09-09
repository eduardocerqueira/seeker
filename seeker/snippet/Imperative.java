//date: 2021-09-09T17:06:54Z
//url: https://api.github.com/gists/b9a54fbfbe1fd3719be50e61fbe359c5
//owner: https://api.github.com/users/kunalgaurav18

List<Integer> numbers = Arrays.asList(new Integer[] {1,2,3,4,5});
List<Integer> result = new ArrayList<>();
for(Integer num : numbers){
  if(num%2 == 0){
    result.add(num);
  }
}
System.out.println(result);
//date: 2022-08-30T17:02:47Z
//url: https://api.github.com/gists/46ade2d8c8e7590dd870ac8e8259a702
//owner: https://api.github.com/users/halitgorgulu

var result = "foo"
  .transform(input -> input + " bar")
  .transform(String::toUpperCase)
System.out.println(result); // FOO BAR

@Test
public void whenTransformUsingParseInt_thenReturnInt() {
    int result = "42".transform(Integer::parseInt);

    assertThat(result, equalTo(42));
}
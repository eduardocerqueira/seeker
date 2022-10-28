//date: 2022-10-28T17:04:24Z
//url: https://api.github.com/gists/ac6aa20266de3ad66dc7d83b1f70942b
//owner: https://api.github.com/users/instancio

Person person = Instancio.of(Person.class)
    .set(all(LocalDateTime.class), LocalDateTime.now())
    .create();
//date: 2022-10-28T17:07:20Z
//url: https://api.github.com/gists/226a64040c42cdd310428bfc5edfb475
//owner: https://api.github.com/users/instancio

Person person = Instancio.of(Person.class)
	.supply(all(LocalDateTime.class), () -> LocalDateTime.now())
	.create();
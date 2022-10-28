//date: 2022-10-28T17:13:10Z
//url: https://api.github.com/gists/c00495487ff394206e180dec96b88556
//owner: https://api.github.com/users/instancio

Person person = Instancio.of(Person.class)
	.generate(all(LocalDateTime.class), gen -> gen.temporal().localDateTime().past())
	.create();
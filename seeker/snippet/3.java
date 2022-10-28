//date: 2022-10-28T17:08:59Z
//url: https://api.github.com/gists/fe2d5cc895801c4ab68ae4ad67d0ed26
//owner: https://api.github.com/users/instancio

LocalDateTime now = LocalDateTime.now();
Person person = Instancio.of(Person.class)
	.supply(all(LocalDateTime.class), () -> now)
	.create();
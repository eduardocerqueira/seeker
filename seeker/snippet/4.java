//date: 2022-10-28T17:12:03Z
//url: https://api.github.com/gists/5ac8c8fd266dcd1427db44b1b1654de8
//owner: https://api.github.com/users/instancio

Person person = Instancio.of(Person.class)
	.supply(all(LocalDateTime.class), random ->
		LocalDateTime.now()
			.minusYears(random.intRange(1, 10))
			.minusDays(random.intRange(1, 365))
			.minusSeconds(random.intRange(1, 60 * 60 * 24))
			.minusNanos(random.intRange(1, 1000000000)))
	.create();
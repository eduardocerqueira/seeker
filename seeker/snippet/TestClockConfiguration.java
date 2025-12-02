//date: 2025-12-02T17:03:03Z
//url: https://api.github.com/gists/cc10b2fd5b5b447c6df1ec90bbddd87a
//owner: https://api.github.com/users/Alex-st

public class OverrideTestConfiguration {

    public static final long DEFAULT_TIME_MILLIS = 1577836800000L;
    public static final Instant DEFAULT_TIME_INSTANT = Instant.ofEpochMilli(DEFAULT_TIME_MILLIS);

    @Bean
    Clock clock() {
        return new TestClock(DEFAULT_TIME_INSTANT);
    }
}
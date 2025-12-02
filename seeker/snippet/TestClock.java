//date: 2025-12-02T17:03:03Z
//url: https://api.github.com/gists/cc10b2fd5b5b447c6df1ec90bbddd87a
//owner: https://api.github.com/users/Alex-st

import lombok.AccessLevel;
import lombok.AllArgsConstructor;

import java.time.Clock;
import java.time.Instant;
import java.time.ZoneId;
import java.time.temporal.ChronoUnit;
import java.util.Objects;

@AllArgsConstructor(access = AccessLevel.PRIVATE)
public class TestClock extends Clock {
    private final Instant defaultInstant;
    private Instant instant;

    public TestClock(Instant instant) {
        Objects.requireNonNull(instant, "instant");
        defaultInstant = instant;
        this.instant = instant;
    }

    public TestClock reset() {
        instant = defaultInstant;
        return this;
    }

    public TestClock setTime(long millis) {
        instant = Instant.ofEpochMilli(millis);
        return this;
    }

    public TestClock plusDays(long days) {
        instant = instant.plus(days, ChronoUnit.DAYS);
        return this;
    }

    public TestClock plusHours(long hours) {
        instant = instant.plus(hours, ChronoUnit.HOURS);
        return this;
    }

    public TestClock plusMinutes(long minutes) {
        instant = instant.plus(minutes, ChronoUnit.MINUTES);
        return this;
    }

    public TestClock plusSeconds(long seconds) {
        instant = instant.plus(seconds, ChronoUnit.SECONDS);
        return this;
    }

    @Override
    public Instant instant() {
        return instant;
    }

    @Override
    public ZoneId getZone() {
        return ZoneId.systemDefault();
    }

    @Override
    public Clock withZone(ZoneId zone) {
        throw new UnsupportedOperationException("Test clock does not support withZone method");
    }
}
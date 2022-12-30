//date: 2022-12-30T16:53:45Z
//url: https://api.github.com/gists/95685480f91970bf6ea2c6ece97c3459
//owner: https://api.github.com/users/williancorrea

public class VersionTest {
    @Test
    public void newInstance_withTwoDotRelease_isParsedCorrectly() {
        final Version version = new Version("1.26.6");
        assertThat(version.numbers, is(new int[]{1, 26, 6}));
    }

    @Test
    public void newInstance_withTwoDotReleaseAndPreReleaseName_isParsedCorrectly() {
        final Version version = new Version("1.26.6-DEBUG");
        assertThat(version.numbers, is(new int[]{1, 26, 6}));
    }

    @Test
    public void compareTo_withEarlierVersion_isGreaterThan() {
        assertThat(new Version("2.0.0").compareTo(new Version("1.0.0")), is(1));
    }

    @Test
    public void compareTo_withSameVersion_isEqual() {
        assertThat(new Version("2.0.0").compareTo(new Version("2.0.0")), is(0));
    }

    @Test
    public void compareTo_withLaterVersion_isLessThan() {
        assertThat(new Version("1.0.0").compareTo(new Version("2.0.0")), is(-1));
    }

    @Test
    public void compareTo_withMorePreciseSameVersion_isFalse() {
        assertThat(new Version("1").compareTo(new Version("1.0.0")), is(0));
    }

    @Test
    public void compareTo_withMorePreciseEarlierVersion_isFalse() {
        assertThat(new Version("2").compareTo(new Version("1.0.0")), is(1));
    }

    @Test
    public void compareTo_withMorePreciseLaterVersion_isLessThan() {
        assertThat(new Version("1").compareTo(new Version("1.0.1")), is(-1));
    }
}

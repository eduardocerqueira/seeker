//date: 2023-11-20T17:08:36Z
//url: https://api.github.com/gists/dafa330947d3ed7b48e7fe0a2cb1f995
//owner: https://api.github.com/users/nadvolod

public class ConditionTesterTest {
    private final ConditionTester tester = new ConditionTester();

    @Test
    public void testBothOptionsWithConditionATrueAndConditionBTrue() {
        assertTrue(tester.option1(true, true));
        assertTrue(tester.option2(true, true));
    }

    @Test
    public void testBothOptionsWithConditionATrueAndConditionBFalse() {
        assertTrue(tester.option1(true, false));
        assertTrue(tester.option2(true, false));
    }

    @Test
    public void testBothOptionsWithConditionAFalse() {
        assertTrue(tester.option1(false, true));
        assertTrue(tester.option2(false, true));
        assertTrue(tester.option1(false, false));
        assertTrue(tester.option2(false, false));
    }
}
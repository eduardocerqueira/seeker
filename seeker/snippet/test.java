//date: 2024-12-04T17:09:52Z
//url: https://api.github.com/gists/3b3fb1487c2b2bba023de8f929030509
//owner: https://api.github.com/users/MichalBrylka

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.time.LocalDate;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;

class BusinessDayCalculatorTest {

    static Stream<Arguments> previousBusinessDayTestData() {
        return Stream.of(
            Arguments.of(LocalDate.of(2024, 12, 2), LocalDate.of(2024, 11, 29)), // Monday -> Friday
            Arguments.of(LocalDate.of(2024, 12, 9), LocalDate.of(2024, 12, 6)),  // Monday -> Friday
            Arguments.of(LocalDate.of(2024, 12, 5), LocalDate.of(2024, 12, 4))   // Thursday -> Wednesday
        );
    }

    static Stream<Arguments> nextBusinessDayTestData() {
        return Stream.of(
            Arguments.of(LocalDate.of(2024, 11, 29), LocalDate.of(2024, 12, 2)), // Friday -> Monday
            Arguments.of(LocalDate.of(2024, 12, 6), LocalDate.of(2024, 12, 9)),  // Friday -> Monday
            Arguments.of(LocalDate.of(2024, 12, 4), LocalDate.of(2024, 12, 5))   // Wednesday -> Thursday
        );
    }

    static Stream<Arguments> previousBusinessDaysTestData() {
        return Stream.of(
            Arguments.of(LocalDate.of(2024, 12, 6), Stream.of(
                LocalDate.of(2024, 12, 5),
                LocalDate.of(2024, 12, 4),
                LocalDate.of(2024, 12, 3),
                LocalDate.of(2024, 12, 2),
                LocalDate.of(2024, 11, 29)
            )),
            Arguments.of(LocalDate.of(2024, 12, 9), Stream.of(
                LocalDate.of(2024, 12, 6),
                LocalDate.of(2024, 12, 5),
                LocalDate.of(2024, 12, 4),
                LocalDate.of(2024, 12, 3),
                LocalDate.of(2024, 12, 2)
            ))
        );
    }

    static Stream<Arguments> nextBusinessDaysTestData() {
        return Stream.of(
            Arguments.of(LocalDate.of(2024, 12, 6), Stream.of(
                LocalDate.of(2024, 12, 9),
                LocalDate.of(2024, 12, 10),
                LocalDate.of(2024, 12, 11),
                LocalDate.of(2024, 12, 12),
                LocalDate.of(2024, 12, 13)
            )),
            Arguments.of(LocalDate.of(2024, 12, 4), Stream.of(
                LocalDate.of(2024, 12, 5),
                LocalDate.of(2024, 12, 6),
                LocalDate.of(2024, 12, 9),
                LocalDate.of(2024, 12, 10),
                LocalDate.of(2024, 12, 11)
            ))
        );
    }

    @ParameterizedTest
    @MethodSource("previousBusinessDayTestData")
    @DisplayName("Test previousBusinessDay method")
    void testPreviousBusinessDay(LocalDate input, LocalDate expected) {
        assertEquals(expected, BusinessDayCalculator.previousBusinessDay(input));
    }

    @ParameterizedTest
    @MethodSource("nextBusinessDayTestData")
    @DisplayName("Test nextBusinessDay method")
    void testNextBusinessDay(LocalDate input, LocalDate expected) {
        assertEquals(expected, BusinessDayCalculator.nextBusinessDay(input));
    }

    @ParameterizedTest
    @MethodSource("previousBusinessDaysTestData")
    @DisplayName("Test previousBusinessDays method")
    void testPreviousBusinessDays(LocalDate input, Stream<LocalDate> expectedStream) {
        assertEquals(expectedStream.toList(), BusinessDayCalculator.previousBusinessDays(input).limit(5).toList());
    }

    @ParameterizedTest
    @MethodSource("nextBusinessDaysTestData")
    @DisplayName("Test nextBusinessDays method")
    void testNextBusinessDays(LocalDate input, Stream<LocalDate> expectedStream) {
        assertEquals(expectedStream.toList(), BusinessDayCalculator.nextBusinessDays(input).limit(5).toList());
    }
}

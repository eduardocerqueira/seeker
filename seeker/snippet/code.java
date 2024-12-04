//date: 2024-12-04T17:09:52Z
//url: https://api.github.com/gists/3b3fb1487c2b2bba023de8f929030509
//owner: https://api.github.com/users/MichalBrylka

import java.time.LocalDate;
import java.util.stream.Stream;

public class BusinessDayCalculator {

    public static LocalDate previousBusinessDay(LocalDate date) {
        LocalDate previous = date.minusDays(1);
        while (isHoliday(previous)) {
            previous = previous.minusDays(1);
        }
        return previous;
    }

    public static Stream<LocalDate> previousBusinessDays(LocalDate date) {
        return Stream.iterate(date.minusDays(1), d -> d.minusDays(1))
                     .filter(d -> !isHoliday(d));
    }

    public static LocalDate nextBusinessDay(LocalDate date) {
        LocalDate next = date.plusDays(1);
        while (isHoliday(next)) {
            next = next.plusDays(1);
        }
        return next;
    }

    public static Stream<LocalDate> nextBusinessDays(LocalDate date) {
        return Stream.iterate(date.plusDays(1), d -> d.plusDays(1))
                     .filter(d -> !isHoliday(d));
    }

    public static boolean isHoliday(LocalDate date) {
        // Example: weekends are holidays
        return date.getDayOfWeek().getValue() >= 6;
    }
}

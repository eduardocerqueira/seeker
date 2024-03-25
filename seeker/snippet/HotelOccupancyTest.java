//date: 2024-03-25T17:00:34Z
//url: https://api.github.com/gists/aafa7e29f1feb8d7a4a3f3ced9f0d301
//owner: https://api.github.com/users/BearsAreAwsome

package edu.citadel;

import org.junit.Test;
import org.junit.runner.RunWith;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;

import static org.junit.Assert.assertEquals;

public class HotelOccupancyTest {
    @Test
    public void calcRateTest1() {
        String inputString = "8 1 18 20 19 9";
        System.setIn(new ByteArrayInputStream(inputString.getBytes()));
        ByteArrayOutputStream systemOut = new ByteArrayOutputStream();
        System.setOut(new PrintStream(systemOut));
        hotelOccupancy.calcRate();
        assertEquals("Enter the number of occupied suites on each of the following floors.\n\n\n" +
                "Floor 10: \n" +
                "\n" +
                "Floor 11: \n" +
                "\n" +
                "Floor 12: \n" +
                "\n" +
                "Floor 14: \n" +
                "\n" +
                "Floor 15: \n" +
                "\n" +
                "Floor 16: \n" +
                "\n" +
                "\n" +
                "The hotel has a total of 120 suites.\n" +
                "\n" +
                "75 are currently occupied.\n" +
                "\n" +
                "This is an occupancy rate of 62% \n\n", systemOut.toString());
    }
    @Test
    public void calcRateTest2() {
        String inputString = "-1 0 21 -3 3 18 20 19 26 17";
        System.setIn(new ByteArrayInputStream(inputString.getBytes()));
        ByteArrayOutputStream systemOut = new ByteArrayOutputStream();
        System.setOut(new PrintStream(systemOut));
        hotelOccupancy.calcRate();
        assertEquals("Enter the number of occupied suites on each of the following floors.\n\n\n" +
                "Floor 10: \n" +
                "\n" +
                "The number of occupied suites must be between 0 and 20\n" +
                "\n" +
                " Re-enter the number of occupied suites on floor 10: \n" +
                "\n" +
                "Floor 11: \n" +
                "\n" +
                "The number of occupied suites must be between 0 and 20\n" +
                "\n" +
                " Re-enter the number of occupied suites on floor 11: \n" +
                "\n" +
                "The number of occupied suites must be between 0 and 20\n" +
                "\n" +
                " Re-enter the number of occupied suites on floor 11: \n" +
                "\n" +
                "Floor 12: \n" +
                "\n" +
                "Floor 14: \n" +
                "\n" +
                "Floor 15: \n" +
                "\n" +
                "Floor 16: \n" +
                "\n" +
                "The number of occupied suites must be between 0 and 20\n" +
                "\n" +
                " Re-enter the number of occupied suites on floor 16: \n" +
                "\n" +
                "\n" +
                "The hotel has a total of 120 suites.\n" +
                "\n" +
                "77 are currently occupied.\n" +
                "\n" +
                "This is an occupancy rate of 64% \n\n", systemOut.toString());
    }
}
//date: 2023-11-16T17:06:54Z
//url: https://api.github.com/gists/28b40cf2a62ba84569c01ec94fbad9ba
//owner: https://api.github.com/users/ndavisc

    package com.codedifferently.flights;

public class Seat {

    private String firstName;
    private String lastName;
    private String email;
    private SeatType type;

    public Seat(String firstName, String lastName, String email, SeatType type){
        this.firstName = firstName;
        this.lastName = lastName;
        this.email = email;
        this.type = type;
    }

    public Seat(SeatType type){
        this.type = type;
        this.firstName = "";
        this.lastName = "";
        this.email = "";
    }

    public String getFirstName(){
        return firstName;
    }

    public void setFirstName(String firstName){
        this.firstName = firstName;
    }

    public String getLastName(){
        return lastName;
    }

    public void setLastName(String lastName){
        this.lastName = lastName;
    }

    public String getEmail(){
        return email;
    }

    public void setEmail(String email){
        this.email = email;
    }

    public SeatType getType(){
        return type;
    }

    public void setType(SeatType type){
        this.type = type;
    }

    public Boolean isAvailable(){
        boolean available = (firstName.equals("") && lastName.equals("") && email.equals(""));
        return available;
    }
    public String toString(){
        StringBuilder builder = new StringBuilder();
        if(firstName.equals("") || lastName.equals("") || email.equals(""))
            return String.format("Passenger: empty; Seat type: %s", type.toString());
        builder.append(String.format("Passenger: %s, %s;",lastName, firstName));
        builder.append(String.format(" email:%s;", email));
        builder.append(" Seat type: " + type.toString());
        return builder.toString();
    }
}



SeatTest.java

package com.codedifferently.flights;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;

public class SeatTest {

    @Test
    public void constructorTest01(){
        // Given
        Seat seat = new Seat("Bob", "TheBuilder", "bob@thebuilder.com", SeatType.FIRST);

        // When
        String expected = "Passenger: TheBuilder, Bob; email:bob@thebuilder.com; Seat type: FIRST";
        String actual = seat.toString();

        // Then
        assertEquals(expected, actual);
    }

    @Test
    public void constructorTest02(){
        Seat seat = new Seat(SeatType.FIRST);
        String expected = "Passenger: empty; Seat type: FIRST";
        String actual = seat.toString();

        assertEquals(expected, actual);
    }

    @Test
    public void getAndSetFirstNameTest01(){
        Seat seat = new Seat("Bob", "TheBuilder", "bob@thebuilder.com", SeatType.FIRST);

        seat.setFirstName("Bobbert");
        String expected = "Bobbert";
        String actual = seat.getFirstName();

        assertEquals(expected, actual);
    }

    @Test
    public void getAndSetLastNameTest01(){
        Seat seat = new Seat("Bob", "TheBuilder", "bob@thebuilder.com", SeatType.FIRST);

        seat.setLastName("TheFlyGuy");
        String expected = "TheFlyGuy";
        String actual = seat.getLastName();

        assertEquals(expected, actual);
    }

    @Test
    public void getAndSetEmailTest01(){
        Seat seat = new Seat("Bob", "TheBuilder", "bob@thebuilder.com", SeatType.FIRST);

        seat.setEmail("bobberttheflyguy@yahoo.com");
        String expected = "bobberttheflyguy@yahoo.com";
        String actual = seat.getEmail();

        assertEquals(expected, actual);
    }

    @Test
    public void getAndSetType01(){
        Seat seat = new Seat("Bob", "TheBuilder", "bob@thebuilder.com", SeatType.ECONOMY);
        seat.setType(SeatType.FIRST);
        SeatType expected = SeatType.FIRST;
        SeatType actual = seat.getType();

        assertEquals(expected, actual);
    }

    @Test
    public void isAvailableTest01(){
        Seat seat = new Seat(SeatType.FIRST);
        Boolean actual = seat.isAvailable();

        assertTrue(actual);
    }

    @Test
    public void isAvailableTest02(){
        Seat seat = new Seat("Bob", "TheBuilder", "bob@thebuilder.com", SeatType.ECONOMY);
        assertFalse(seat.isAvailable());
    }

    // Todo: Completing all the getters and setters for the class with unit test
}


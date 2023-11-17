//date: 2023-11-17T16:29:58Z
//url: https://api.github.com/gists/e91a8079aca180deac3b9ec1b12ea1d4
//owner: https://api.github.com/users/TheLordLulu

package com.codedifferently.lab;

public class Watch {
    private String brand;
    private String material;
    private int waterResistantDepth;
    private  boolean hasAlarm;

    public Watch(String brand, String material, int waterResistantDepth, boolean hasAlarm) {
        this.brand = brand;
        this.material = material;
        this.waterResistantDepth = waterResistantDepth;
        this.hasAlarm = hasAlarm;
    }

    public String getBrand() {
        return brand;
    }

    public void setBrand(String brand) {
        this.brand = brand;
    }

    public String getMaterial() {
        return material;
    }

    public void setMaterial(String material) {
        this.material = material;
    }

    public int getWaterResistantDepth() {
        return waterResistantDepth;
    }

    public void setWaterResistantDepth(int waterResistantDepth) {
        this.waterResistantDepth = waterResistantDepth;
    }

    public boolean getHasAlarm() {
        return hasAlarm;
    }


    public void setHasAlarm(boolean hasAlarm) {
        this.hasAlarm = hasAlarm;
    }

    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append(String.format("Brand: %s%n", brand));
        builder.append(String.format("Material: %s%n", material));
        builder.append(String.format("Water Resistant Depth: %s%n", waterResistantDepth));
        builder.append(String.format("Has Alarm: %s%n", hasAlarm));

        return builder.toString();
    }
}

package com.codedifferently.lab;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class WatchTest {

    @Test
    public void laptopTestConstructor(){
        //Given
        Watch watch = new Watch("Apple","Leather", 256, true);

        //When
        String expected =
                """
                        Brand: Apple
                        Material: Leather
                        Water Resistant Depth: 256
                        Has Alarm: true
                        """;
        String actual = watch.toString();

        //Then
        assertEquals(expected, actual);
    }

    @Test
    public void getAndSetBrand(){
        Watch watch = new Watch("Apple","Leather", 256, true);

        watch.setBrand("Apple");
        String expected = "Apple";
        String actual = watch.getBrand();

        assertEquals(expected, actual);
    }

    @Test
    public void getAndSetMaterial(){
        Watch watch = new Watch("Apple","Leather", 256, true);

        watch.setMaterial("Leather");
        String expected = "Leather";
        String actual = watch.getMaterial();

        assertEquals(expected, actual);
    }

    @Test
    public void getAndSetWaterResistantDepth(){
        Watch watch = new Watch("Apple","Leather", 256, true);

        watch.setWaterResistantDepth(256);
        int expected = 256;
        int actual = watch.getWaterResistantDepth();

        assertEquals(expected, actual);
    }


    @Test
    public void getAndSetHasAlarm(){
        Watch watch = new Watch("Apple","Leather", 256, true);

        watch.setHasAlarm(true);
        boolean expected = true;
        boolean actual = watch.getHasAlarm();

        assertEquals(expected, actual);
    }
}



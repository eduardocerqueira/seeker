//date: 2023-07-24T16:46:36Z
//url: https://api.github.com/gists/4ac35f6386534ae80821b46a0a4e1673
//owner: https://api.github.com/users/jairoArh

package generic;

import java.util.Comparator;

public class Dish  {

    private String id;
    private String name;
    private double value;

    private boolean isVegetarian;

    private int calories;

    public Dish(String id, String name, double value, boolean isVegetarian, int calories) {
        this.id = id;
        this.name = name;
        this.value = value;
        this.isVegetarian = isVegetarian;
        this.calories = calories;
    }

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public double getValue() {
        return value;
    }

    public void setValue(double value) {
        this.value = value;
    }

    public boolean isVegetarian() {
        return isVegetarian;
    }

    public void setVegetarian(boolean vegetarian) {
        isVegetarian = vegetarian;
    }

    public int getCalories() {
        return calories;
    }

    public void setCalories(int calories) {
        this.calories = calories;
    }

    @Override
    public String toString() {
        return "Dish{" +
                "id='" + id + '\'' +
                ", name='" + name + '\'' +
                ", value=" + value +
                ", isVegetarian=" + isVegetarian +
                ", calories=" + calories +
                '}';
    }

}

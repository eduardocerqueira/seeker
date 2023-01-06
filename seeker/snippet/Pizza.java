//date: 2023-01-06T16:50:41Z
//url: https://api.github.com/gists/cd88cede93475ad6617bb3d91d027cc6
//owner: https://api.github.com/users/frchiron

package org.domain;

import java.util.List;

public class Pizza {

    public enum Topping {PEPERRONI, CHORIZO, HAM, OLIVES, SALMON, EGG, MUSHROOMS;};

    public enum Dough {THIN, FAT};

    public enum Base {TOMATE_SAUCE, CREAM};

    public enum Cheese {MOZZARELLA, CHEDDAR, COMTE, EMMENTAL};

    private final String name;
    private Dough dough;
    private Base base;
    private List toppings;
    private List cheeses;

    public Pizza(PizzaStepBuilder.Steps steps) {
        this.name = steps.name;
        this.base = steps.base;
        this.dough = steps.dough;
        this.toppings = steps.toppings;
        this.cheeses = steps.cheeses;
    }

    public String getName() {
        return name;
    }

    public Dough getDough() {
        return dough;
    }

    public Base getBase() {
        return base;
    }

    public List getToppings() {
        return toppings;
    }

    public List getCheeses() {
        return cheeses;
    }

    @Override
    public String toString() {
        return "Pizza{" +
                "name='" + name + '\'' +
                ", dough='" + dough + '\'' +
                ", base='" + base + '\'' +
                ", toppings=" + toppings +
                ", cheeses=" + cheeses +
                '}';
    }
}

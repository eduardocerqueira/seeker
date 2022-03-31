//date: 2022-03-31T17:10:03Z
//url: https://api.github.com/gists/ff486ad0c1510321bbe08bb6d02c4605
//owner: https://api.github.com/users/Suranchiyev

import java.util.ArrayList;
import java.util.Arrays;

public class Main {
    public static void main(String[] args) {
        PizzaBuilder pizzaBuilder = new PizzaBuilder();
        // use method chaining to build properties
        pizzaBuilder.setName("Veggie").setPrice(15.99).setToppings(new ArrayList<>(Arrays.asList("broccoli", "squash", "tomatoes")));
        pizzaBuilder.setSize(11).setTips(4.00);

        // get Pizza object with final build method
        Pizza pizza = pizzaBuilder.build();
        System.out.println(pizza.getName());
        System.out.println(pizza.getPrice());
        System.out.println(pizza.getToppings());
        System.out.println(pizza.getSize());
        System.out.println(pizza.getTips());
        System.out.println(pizza.getDeliveryAddress());
        System.out.println("-----------------");

        
        // build with few properties only
        Pizza pizza2 = new PizzaBuilder().setName("Chicago Pizza").setPrice(21.50).build();
        System.out.println(pizza2.getName());
        System.out.println(pizza2.getPrice());
        System.out.println(pizza2.getToppings());
    }
}

class PizzaBuilder {
    private String name;
    private double price;
    private ArrayList<String> toppings = new ArrayList<>();
    private int size;
    private String deliveryAddress;
    private String storeAddress;
    private String storeManager;
    private double tips;

    // setters that set the value and return instance of PizzaBuilder
    // so they can be used through method chaining
    public PizzaBuilder setName(String name) {
        this.name = name;
        return this;
    }

    public PizzaBuilder setPrice(double price) {
        this.price = price;
        return this;
    }

    public PizzaBuilder setToppings(ArrayList<String> toppings) {
        this.toppings = toppings;
        return this;
    }

    public PizzaBuilder setSize(int size) {
        this.size = size;
        return this;
    }

    public PizzaBuilder setDeliveryAddress(String deliveryAddress) {
        this.deliveryAddress = deliveryAddress;
        return this;
    }

    public PizzaBuilder setStoreAddress(String storeAddress) {
        this.storeAddress = storeAddress;
        return this;
    }

    public PizzaBuilder setStoreManager(String storeManager) {
        this.storeManager = storeManager;
        return this;
    }

    public PizzaBuilder setTips(double tips) {
        this.tips = tips;
        return this;
    }

    // build method that return immutable Pizza
    public Pizza build() {
        return new Pizza(name, price, toppings, size, deliveryAddress, storeAddress, storeManager, tips);
    }
}

// Immutable Pizza
final class Pizza {
    private final String name;
    private final double price;
    private final ArrayList<String> toppings;
    private final int size;
    private final String deliveryAddress;
    private final String storeAddress;
    private final String storeManager;
    private final double tips;

    public Pizza(String name, double price, ArrayList<String> toppings,
                 int size, String deliveryAddress, String storeAddress,
                 String storeManager, double tips) {
        this.name = name;
        this.price = price;
        this.toppings = new ArrayList<>(toppings);
        this.size = size;
        this.deliveryAddress = deliveryAddress;
        this.storeAddress = storeAddress;
        this.storeManager = storeManager;
        this.tips = tips;
    }

    public String getName() {
        return name;
    }

    public double getPrice() {
        return price;
    }

    public ArrayList<String> getToppings() {
        return new ArrayList<>(toppings);
    }

    public int getSize() {
        return size;
    }

    public String getDeliveryAddress() {
        return deliveryAddress;
    }

    public String getStoreAddress() {
        return storeAddress;
    }

    public String getStoreManager() {
        return storeManager;
    }

    public double getTips() {
        return tips;
    }
}



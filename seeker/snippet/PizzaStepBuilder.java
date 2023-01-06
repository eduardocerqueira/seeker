//date: 2023-01-06T16:50:41Z
//url: https://api.github.com/gists/cd88cede93475ad6617bb3d91d027cc6
//owner: https://api.github.com/users/frchiron

package org.domain;

import java.util.ArrayList;
import java.util.List;

public class PizzaStepBuilder {

    public static NameStep newBuilder() {
        return new Steps();
    }

    private PizzaStepBuilder() {
    }

    public static interface NameStep {
        DoughStep pizzaName(String name);
    }

    public static interface DoughStep {
        BaseStep withDough(Pizza.Dough dough);
    }

    public static interface BaseStep {
        ToppingStep withBase(Pizza.Base base);
    }

    public static interface ToppingStep {
        ToppingStep addTopping(Pizza.Topping topping);

        CheeseStep noToppingsPlease();

        CheeseStep noMoreToppingsPlease();
    }

    public static interface CheeseStep {
        CheeseStep addCheese(Pizza.Cheese cheese);

        BuildStep noCheesePlease();

        BuildStep noMoreCheesePlease();
    }

    public static interface BuildStep {
        Pizza build();
    }

    static class Steps implements NameStep, DoughStep, BaseStep, ToppingStep, CheeseStep, BuildStep {
        String name;
        final List toppings = new ArrayList<>();
        final List cheeses = new ArrayList<>();
        Pizza.Dough dough;
        Pizza.Base base;

        @Override
        public DoughStep pizzaName(String name) {
            this.name = name;
            return this;
        }

        @Override
        public BaseStep withDough(Pizza.Dough dough) {
            this.dough = dough;
            return this;
        }

        @Override
        public ToppingStep withBase(Pizza.Base base) {
            this.base = base;
            return this;
        }

        @Override
        public ToppingStep addTopping(Pizza.Topping topping) {
            this.toppings.add(topping);
            return this;
        }

        @Override
        public CheeseStep noToppingsPlease() {
            return this;
        }

        @Override
        public CheeseStep noMoreToppingsPlease() {
            return this;
        }

        @Override
        public CheeseStep addCheese(Pizza.Cheese cheese) {
            this.cheeses.add(cheese);
            return this;
        }

        @Override
        public BuildStep noCheesePlease() {
            return this;
        }

        @Override
        public BuildStep noMoreCheesePlease() {
            return this;
        }

        @Override
        public Pizza build() {
            return new Pizza(this);
        }
    }

}

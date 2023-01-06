//date: 2023-01-06T16:50:41Z
//url: https://api.github.com/gists/cd88cede93475ad6617bb3d91d027cc6
//owner: https://api.github.com/users/frchiron

package org;

import org.domain.Pizza;
import org.domain.PizzaStepBuilder;

import java.util.List;

import static org.domain.Pizza.*;

/**
 * Hello world!
 */
public class App {
    public static void main(String[] args) {

        Pizza reineAgain = PizzaStepBuilder.newBuilder()
                .pizzaName("Marguerita")
                .withDough(Dough.THIN)
                .withBase(Base.TOMATE_SAUCE)
                .addTopping(Topping.HAM)
                .addTopping(Topping.MUSHROOMS)
                .noMoreToppingsPlease()
                .addCheese(Cheese.MOZZARELLA)
                .noMoreCheesePlease()
                .build();

        System.out.println("Pizza : " + reineAgain);

        /*
        // cannot build weird pizza
        Pizza tryWeirdPizza = PizzaStepBuilder.newBuilder()
                .pizzaName("Super Weird Pizza")
                .withDough(Dough.THIN)
                .
        */
        
    }
}

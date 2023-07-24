//date: 2023-07-24T16:46:36Z
//url: https://api.github.com/gists/4ac35f6386534ae80821b46a0a4e1673
//owner: https://api.github.com/users/jairoArh

package generic;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Runner {
    public static void main(String[] args) {

        HandlingDish<Dish> hd =  new HandlingDish<>(((o1, o2) -> o1.getName().compareTo(o2.getName())));

        System.out.println(hd.addDish(new Dish("13","Carne",35000,false,450)));
        System.out.println(hd.addDish(new Dish("75","Pescado",28000,false,380)));
        System.out.println(hd.addDish(new Dish("24","Pollo",34000,true,100)));
        System.out.println(hd.addDish(new Dish("35","Fruta",19000,true,50)));
        System.out.println(hd.addDish(new Dish("645","Hamburgeuesa",19000,true,50)));

        //hd.getDishes().forEach( System.out::println );


        List<Integer> nums = Arrays.asList(4,8,44);

        List<Dish> dishes = List.of(new Dish("543","Tomahowk",89000,false,700),new Dish("100","Bagre en Salsa",45000,true,200));

        dishes.stream().sorted( ((o1, o2) -> o1.getId().compareTo(o2.getId())))
                .forEach( System.out::println);



    }
}

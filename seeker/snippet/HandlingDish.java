//date: 2023-07-24T16:46:36Z
//url: https://api.github.com/gists/4ac35f6386534ae80821b46a0a4e1673
//owner: https://api.github.com/users/jairoArh

package generic;

import java.util.*;

public class HandlingDish<T> {
    private ArrayList<T> list;

    private Comparator<T> comparator;

    public HandlingDish(Comparator<T> comparator) {
        list = new ArrayList<>();
        this.comparator = comparator;
    }

    public T findObject( T object ){
        for( T t : list ){
            if( comparator.compare( t, object) == 0){
                return t;
            }
        }

        return null;
    }

    public boolean addDish(T object){
        if( findObject( object ) == null ){

            list.add( object );

            return true;
        }

        return false;
    }


    public ArrayList<Dish> getDishes() {
        return (ArrayList<Dish>) list.clone();
    }

    public ArrayList<Dish> sort( Comparator<Dish> comparator){

        ArrayList<Dish> sorted = getDishes();
        Collections.sort(sorted, comparator);

        return sorted;
    }
}

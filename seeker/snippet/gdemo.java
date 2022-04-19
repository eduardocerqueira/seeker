//date: 2022-04-19T16:48:03Z
//url: https://api.github.com/gists/e2e983c1f9c2595f4f377cd7d2a38d19
//owner: https://api.github.com/users/Pranjali4jain

import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;
import java.util.Scanner;

public class gdemo<bidirectional> {
    Map<Integer, LinkedList<Integer>> obj;

    public gdemo() {
        obj = new HashMap<>();
    }
    public void addEdge(int v1, int v2, boolean isbidirectional)
    {
        LinkedList<Integer>v1neighbour = obj.getOrDefault(v1,new LinkedList());
        v1neighbour . add(v2);
        obj.put(v1,v1neighbour);

        if(isbidirectional == true)
        {
            LinkedList<Integer>v2neighbour = obj.getOrDefault(v2,new LinkedList());
            v2neighbour . add(v1);
            obj.put(v2,v2neighbour);

        }

    }
    public void display() {
        for (Map.Entry<Integer, LinkedList<Integer>> res : obj.entrySet()
        ) {
            System.out.print(res.getKey() + " ");
            System.out.println(res.getValue());
        }
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        boolean abc = false;
        gdemo obj1 = new gdemo();
        obj1.addEdge(sc.nextInt(), sc.nextInt(), abc);
        obj1.addEdge(sc.nextInt(), sc.nextInt(), abc);
        obj1.display();
    }
}


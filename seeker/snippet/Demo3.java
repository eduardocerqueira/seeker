//date: 2023-07-17T16:47:44Z
//url: https://api.github.com/gists/7360ac6571fbfc73da26342558add862
//owner: https://api.github.com/users/yash8917

import java.util.Enumeration;
import java.util.Vector;
// advantage -> synchronization , Dynamic in nature

public class Demo3 {

    public static void main(String[] args) {
        Vector<Integer> obj = new Vector<>();
        obj.addElement(1);
        obj.addElement(2);
        obj.addElement(3);
        obj.addElement(4);

        // print using Lamda
        obj.forEach(o -> System.out.println(o));

        System.out.println("---------- Using Enumration ---------");
        //enumration only impliments in Vector class
        Enumeration en = obj.elements();
        while(en.hasMoreElements()){
            System.out.println(en.nextElement());
        }


     }
}

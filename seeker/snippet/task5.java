//date: 2023-04-18T16:54:40Z
//url: https://api.github.com/gists/566dd628eb43e1f6a8069715da76c22f
//owner: https://api.github.com/users/alexcreed764

package HomeWork5;

import java.util.LinkedList;
import java.util.Objects;
import java.util.Scanner;
import java.util.Stack;

public class tasks {
    public static void main(String[] args) 
    {
        
    } //END MAIN
    public static void дз5()
    {
        int a; // если заменить на doble бедет считать точнее ;)
        int b;
        int res = 0;
        String op;
        Scanner sc = new Scanner(System.in);
        a = inputNum("Введите 1е число: ");
        b = inputNum("Введите 2е число: ");
        op = inputStr("\nВведите оператор (+, -, *, /): ");
        Stack<Integer> stack = new Stack<>();
        if (Objects.equals(op, "+"))     {res = a + b;}
        else if(Objects.equals(op, "-")) {res = a - b;}
        else if(Objects.equals(op, "*")) {res = a * b;}
        else if(Objects.equals(op, "/")) {res = a / b;}
        else System.out.printf("Неверный ввод !");
        
        System.out.println("ответ:" + res);

        stack.push(res);
        while (true)
        {
            op = inputStr("\n> Введите оператор (+, -, *, /), " + 
            "\n> Либо ведите \"no\" для" +
            "удаление последнего результата(oper или no) ");
            if (Objects.equals(op, "no"))
            {
                System.out.println("удаленное значение: " + stack.pop());
                res = stack.peek(); 
                System.out.println("Текущее значение: " + res);              
            }
            else
            {
                b = inputNum("Введите число: ");
                if (Objects.equals(op, "+"))     {res = res + b;}
                else if(Objects.equals(op, "-")) {res = res - b;}
                else if(Objects.equals(op, "*")) {res = res * b;}
                else if(Objects.equals(op, "/")) {res = res / b;}
                else System.out.printf("Неверный ввод !");
                stack.push(res);
                System.out.println("ответ = " + res);
                
                


            }    
                    
  
        }
    }
    public static int inputNum(String text)
    {
        Scanner scan = new Scanner(System.in);
        System.out.print(text);
        int num = scan.nextInt();
        //scan.close(); //не могу закрыть ошибка !!!
        return num;
    }
    public static String inputStr(String text)
    {
        Scanner scan = new Scanner(System.in);
        System.out.print(text);
        String str = scan.nextLine();
        return str;
    }

    public static void Zadacha1()
    {
        //Пусть дан LinkedList с несколькими элементами. Реализуйте метод(не void),
        //который вернет “перевернутый” список.
        LinkedList<Integer> linList = new LinkedList<>();
        for (int i = 0; i < 6; i++) 
        {
            linList.add(i,i); // тестить это ззначение arList.add(0,i);
        }
        linList.add(6); // просто проверка
        linList.add(7);
        System.out.println("Исходный список: " +linList);    
        System.out.println("Итоговый список: " + reverse(linList)); // метод reverse(linList)


    }
    public static LinkedList<Integer> reverse (LinkedList<Integer> listInit)
    {
        LinkedList<Integer> resList = new LinkedList<>();
        for (int i = listInit.size()-1; i > -1; i--) {
            resList.add(listInit.get(i));
        }
        return resList;
    }

    public static void Zadacha2()
    {
        //Реализуйте очередь с помощью LinkedList со следующими методами:
        //enqueue() - помещает элемент в конец очереди,
        //dequeue() - возвращает первый элемент из очереди и удаляет его,
        //first() - возвращает первый элемент из очереди, не удаляя.

        LinkedList<Integer> linList = new LinkedList<>();
        linList.add(1); 
        linList.add(2);
        linList.add(3);
        System.out.println(linList);
        enqueue(linList, 7); // помещает элемент в конец очереди  и выводит в консоль
        dequeue(linList);         // возвращает первый элемент из очереди, удаляет его и выводит в консоль список
        System.out.println(first(linList)); // возвращает первый элемент из очереди, не удаляя и выводит его в консоль
        
        //linList.get(0);   //  возвращает первый элемент
        //linList.remove(0); // и удаляет его

        


    }
    public static LinkedList<Integer> enqueue (LinkedList<Integer> list, int num) 
    {
        list.addLast(num);
        System.out.println(list);   
        return list;
    } 
    public static int dequeue (LinkedList<Integer> list) 
    {
        int temp = list.get(0);
        list.remove(0);
        System.out.println(list);
        return temp;
    } 
    public static int first (LinkedList<Integer> list) 
    {
        return list.get(0);
    }
}

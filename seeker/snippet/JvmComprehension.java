//date: 2023-04-20T17:02:14Z
//url: https://api.github.com/gists/5ea60295570de53d2c8765fdb2c07543
//owner: https://api.github.com/users/LoneDog001

public class JvmComprehension {
    // ^metaspace
    // После объявления нового класса он загружается в систему с помощью ClassLoader

    public static void main(String[] args) {
        //        ^ Stack Memory
        int i = 1; // 1 Stack Memory
        Object o = new Object(); //2 в heap выделяется память для "о", в Stack Memory сохрагяется ссылка на эту область
        Integer ii = 2;                 // 3 Stack Memory
        printAll(o, i, ii);             // 4 printAll помещается в Stack Memory, куда записываются ссылки на "о" и i, ii
        System.out.println("finished"); // 7 Создастся новый фрейм в Stack Memory, а в heap образуется String со значением "finished"
        //ссылка на него помещается в Stack Memory. После вызова метода и после обработки удаляется из Stack Memory

    }

    /*
          При завершении программы из Stack Memory удаляется фрейм, связанный с main. Оставшиеся в heap объекты ("о" и два String) удаляются сборщиком мусора.
        */
    private static void printAll(Object o, int i, Integer ii) {
        Integer uselessVar = 700;                   // 5 Stack Memory
        System.out.println(o.toString() + i + ii);  // 6 в Stack Memory создается фрейм для метода o.toString(), а в heap выделяется памать под переменную
        // String, представляющую результат этого метода. Вызывается метод и фрейм удаляется из Stack Memory.
        // Далее в Stack Memory выделяется фрейм для System.out.println, где сохраняется ссылка на результат
        // o.toString() и значения i и ii. После отработки метода соответствующий фрейм удаляется из Stack Memory.
    }
}

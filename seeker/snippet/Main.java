//date: 2022-12-16T16:56:49Z
//url: https://api.github.com/gists/5d6fa4295d1bd55afb3ed506be1ddc3e
//owner: https://api.github.com/users/Mikaeryu

import java.math.*;
/*      Реализуйте метод, возвращающий ответ на вопрос: правда ли, что a + b = c?
        Допустимая погрешность – 0.0001 (1E-4)

        Можно использовать класс Math и его методы. Класс Math доступен всегда, импортировать его не надо.

        В качестве примера написано заведомо неправильное выражение. Исправьте его.
*/
public class Main {
    public static void main(String[] args) {
        double a = 0.1;
        double b = 0.2;
        double c = 0.3;
 
        System.out.println(Math.abs(a + b) - c < 1E-4);
    }
}
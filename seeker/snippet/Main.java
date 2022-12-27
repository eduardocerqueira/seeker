//date: 2022-12-27T16:38:00Z
//url: https://api.github.com/gists/fd8291daad05e006ae2d244b4edb894a
//owner: https://api.github.com/users/Mikaeryu

/* Задание 3.4.9
Дан класс ComplexNumber. Переопределите в нем методы equals() и hashCode() так, чтобы equals()
сравнивал экземпляры ComplexNumber по содержимому полей re и im, а hashCode() был бы согласованным с реализацией equals().
Реализация hashCode(), возвращающая константу или не учитывающая дробную часть re и im, засчитана не будет
Пример:
ComplexNumber a = new ComplexNumber(1, 1);
ComplexNumber b = new ComplexNumber(1, 1);
// a.equals(b) must return true
// a.hashCode() must be equal to b.hashCode()
 */
package org.stepic.tasks;

public class Main {
    public static void main(String[] args) {
        ComplexNumber a = new ComplexNumber(1, 1);
        ComplexNumber b = new ComplexNumber(1, 1);

        System.out.println("a: " + "re: " + a.getRe() + " im: " + a.getIm());
        System.out.println("b: " + "re: " + b.getRe() + " im: " + b.getIm());

        System.out.println(a.equals(b));
        System.out.println(a.hashCode() == b.hashCode());
        System.out.println("a: " + a.hashCode() + " b: " + b.hashCode());
        // a.equals(b) must return true
        // a.hashCode() must be equal to b.hashCode()
    }
}
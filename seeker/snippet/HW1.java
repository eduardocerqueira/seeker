//date: 2023-01-17T16:46:21Z
//url: https://api.github.com/gists/da3e7a74b5b079b60b432623ec0be252
//owner: https://api.github.com/users/AStr-ov

// Нужно написать в калькуляторе метод вычисления суммы покупки со скидкой и проверить его, используя AssertJ

import static org.assertj.core.api.Assertions.*;

public class HW1 {
    public static void main(String[] args) {
        calculatingDiscount(200, 25);
        
        // Test();
    }

    public static void Test() {
        assertThat(calculatingDiscount(100, 10)).isEqualTo(90);// Положительная проверка
        assertThat(calculatingDiscount(2000, 120));                    //Проверка скидки > 100%
        assertThat(calculatingDiscount(-100, 25));                     //Проверка отрицательной цены
        assertThat(calculatingDiscount(100, -25));                     //Проверка отрицательной скидки
        assertThat(calculatingDiscount(0, 10));                        //Проверка нулевой цены
    }

    public static double calculatingDiscount(double price, double discount) {
        double res;
        if (discount > 100) {
            throw new ArithmeticException("Скидка не может быть > 100%");
        } else if (price <= 0) {
            throw new ArithmeticException("Цена должна быть > 0");
        } else if (discount < 0) {
            throw new ArithmeticException("Скидка не может быть < 0");
        } else
            res = price * (100 - discount) / 100;
        System.out.println("Сумма " + price + " со скидкой " + discount + " %  составляет - " + res);
        return res;
    }
}

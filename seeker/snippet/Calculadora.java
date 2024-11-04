//date: 2024-11-04T16:56:58Z
//url: https://api.github.com/gists/8be7bd8c8c1faafb1361369bf1eaaa2a
//owner: https://api.github.com/users/lorranetorresx

public class Calculadora {
    public double somar(double a, double b) {
        return a + b;
    }
    public double subtrair(double a, double b) {
        return a - b;
    }
    public double multiplicar(double a, double b) {
        return a * b;
    }
    public double dividir(double a, double b) throws ArithmeticException {
        if (b == 0) {
            throw new ArithmeticException("Divisão a zero não permitida!");
        }
        return a / b;
    }
}

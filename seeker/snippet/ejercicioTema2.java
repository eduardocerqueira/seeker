//date: 2022-09-12T17:14:22Z
//url: https://api.github.com/gists/af70d57410349f86252c8eb8e68a02a9
//owner: https://api.github.com/users/DKasius

package com.Java.Funciones2;

public class ejercicioTema2 {


    public static void main(String[] args) {


        double precio = 250;
        double iva = (precio / 100) * 21;
        double resultado = precio + iva;

        System.out.println("El precio del producto con el iva es: " + resultado + " euros");
    }

    static double getPrice(double precio, double iva, double resultado) {
        return resultado;

    }
}

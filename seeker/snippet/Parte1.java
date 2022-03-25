//date: 2022-03-25T16:55:58Z
//url: https://api.github.com/gists/b7c99b18dfc8b7e798f213798a246530
//owner: https://api.github.com/users/eidarra

package com.company;

public class Main {

    public static void main(String[] args) {
	int resultado;
    resultado = suma(20 , 1 ,2);
    System.out.println(resultado);
    }
    public static int suma(int a, int b, int c){
        return a + b + c;
    }
}
//////Parte 2
public class Main {
    public static void main(String[] args) {
        Coche miCoche = new Coche();
        miCoche.IncrementarPuerta();
        System.out.println(miCoche.puertas);
    }
}
    class Coche {
        public int puertas = 4;
        public void IncrementarPuerta() {
            this.puertas++;
        }
    }

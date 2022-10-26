//date: 2022-10-26T17:08:21Z
//url: https://api.github.com/gists/f8789b0d31256dc695290d0ac9ad9256
//owner: https://api.github.com/users/Jother043

package relación_ejercicios_3_Arrays;

public class ejercicio9_3_3 {


    public static int[] posicionOcupa(int[] numeros) {


        int[] arrReverse = new int[numeros.length];
        int[] ordenadoMenor = new int[numeros.length];
        int numero = 0;
        for (int i = numeros.length; i > 0; i--) {
            arrReverse[i] = numeros[i];
        }
        for (int j = 0; j < arrReverse.length; j++) {
            if (numeros[j] > arrReverse[j]) {
                ordenadoMenor[j] = numeros[j];
            } else {
                ordenadoMenor[j] = numeros[j];
            }
        }

        return ordenadoMenor;
    }

    public static void main(String[] args) {

        int[] array = {1, 2, 3, 4, 5, 8, 9, 7};

        System.out.println(posicionOcupa(array));

    }
}

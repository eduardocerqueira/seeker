//date: 2022-10-26T17:08:36Z
//url: https://api.github.com/gists/76ab27e85ff1390864374e9c86908a4e
//owner: https://api.github.com/users/Jother043

package relación_ejercicios_3_Arrays;

import java.util.Arrays;

public class ejercicio10_3_3 {

    private static int[] eliminaDuplicados(int[] numeros) {

        int[] arrayCribado = new int[numeros.length]; //Creamos un array en el que cribamos los números repetido.
        int numero = 0; //Variable contadora que utilizamos para contar los números que han sido igual.

        for (int i : numeros) { //recorremos numeros.

            if (!buscado(arrayCribado, i)) { //Si buscado
                arrayCribado[numero++] = i;
            }
        }

        if (buscado(numeros, 0)) {
            numero++;
        }

        return recortaArray(arrayCribado, numero);
    }

    private static int[] recortaArray(int[] recortado, int nuevaLongitud) {

        int[] arrayRecortado = new int[nuevaLongitud];
        for (int i = 0; i < nuevaLongitud; i++) {
            arrayRecortado[i] = recortado[i];
        }

        return arrayRecortado;
    }

    private static boolean buscado(int[] array, int numBuscado) {

        boolean encontrado = false;

        for (int i = 0; i < array.length && !encontrado; i++) {

            if (array[i] == numBuscado) {
                encontrado = true;
            }
        }

        return encontrado;

    }

    public static void main(String[] args) {

        int[] array = {1, 2, 2, 4, 5, 4, 3, 7, 7};

        System.out.println(Arrays.toString(eliminaDuplicados(array)));

    }
}
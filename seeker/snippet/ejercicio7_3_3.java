//date: 2022-10-26T17:07:52Z
//url: https://api.github.com/gists/81c1246c9e24a48ccc139f64c2df99cb
//owner: https://api.github.com/users/Jother043

package relación_ejercicios_3_Arrays;

public class ejercicio7_3_3 {



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

        int[] arr = {5, 6, 8, 4, 2, 1, 7, 8, 9};

        if (buscado(arr,3 )) {
            System.out.println("se ha encontrado en la lista");
        } else {
            System.out.println(" No se ha encontrado ");
        }
    }
}

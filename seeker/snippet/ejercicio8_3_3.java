//date: 2022-10-26T17:08:05Z
//url: https://api.github.com/gists/b62e9b355385dca82bddc444a99de2dd
//owner: https://api.github.com/users/Jother043

package relación_ejercicios_3_Arrays;

public class ejercicio8_3_3 {


    private static int posicionOcupa(int[] array, int numeroElegido){
        int posicion = 0;
        for(int i = 0; i < array.length && posicion<=0; i++){

            int posiciones = array[i];

            if(numeroElegido == posiciones){
                posicion = i;
            }else {
                posicion = -1;
            }
        }

        return posicion;
    }
    public static void main(String[] args) {

        int[] numeros = {1,2,3,4,5,8,9,7};
        int posicionOcupada = posicionOcupa(numeros, 6);

        if(posicionOcupada < 0){
            System.out.printf("El numero se encuentra en la posición: %d , por lo tanto no aparece. ", posicionOcupada);
        }else{
            System.out.printf("El numero se encuentra en la posición: %d: ", posicionOcupada);
        }

    }
}

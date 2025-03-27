//date: 2025-03-27T17:06:16Z
//url: https://api.github.com/gists/a740a2d12becfaa0817c833a646df4e3
//owner: https://api.github.com/users/ClaudiaGCastaneda

import java.util.Scanner;

public class ArraysExcercises {
    public static void main(String[] args) {
        
        //Agregar los elementos de una amtriz a otra matriz en orden inverso
        int[] matriz = {1, 2, 3, 4, 5};
        int[] matriz2 =  new int[matriz.length];
        int contador =  matriz.length - 1;

        // for (int i = 0; contador >= 0; i++) {
        //     matriz2[i]  = matriz[contador];
        //     contador--;
        //     System.out.println("Agregnado a matriz2 "  + matriz2[i] );
        // }


        int[][] mat3 =  new int[3][3];

        Scanner scan = new Scanner(System.in);

        for (int i = 0; i < mat3.length; i++) {
            for (int j = 0; j < mat3.length; j++) {
                System.out.println("Proporciona el valor del elemento [" + i + "][" + j + "]");
                mat3[i][j] =  scan.nextInt();
                
            }
            
        }

        scan.close();

        for (int i = 0; i < mat3.length; i++) {
            for (int j = 0; j < mat3.length; j++) {
                System.out.print(mat3[i][j] + " ");
                
            }
            System.out.println();
        }


    }
}

//date: 2025-12-24T17:02:25Z
//url: https://api.github.com/gists/00437ee42f4ba1e29b3c090d1ca808ce
//owner: https://api.github.com/users/DiscoDurodeRoer

package com.mycompany.stack;

import java.util.Stack;

public class ExampleStack {

    public static void main(String[] args) {

        // Creamos una pila de numeros
        Stack<Integer> stack = new Stack<>();

        // Añadimos elementos a la pila
        stack.push(5); // 5
        stack.push(10); // 10 5
        stack.push(15); // 15 10 5
        
        // Recorrer la pila vaciandola
        while(!pila.isEmpty()){
            System.out.println("Elemento extraido: " + stack.pop());
        }

    }
}
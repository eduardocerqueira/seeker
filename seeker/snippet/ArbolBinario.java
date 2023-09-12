//date: 2023-09-12T16:55:48Z
//url: https://api.github.com/gists/fc2d0e0399df5bbbf30b678a81e9a9e2
//owner: https://api.github.com/users/Nasheli5U

import java.util.LinkedList;
import java.util.Queue;

class Nodo {
    int valor;
    Nodo izquierda, derecha;

    public Nodo(int item) {
        valor = item;
        izquierda = derecha = null;
    }
}

public class ArbolBinario {
    Nodo raiz;

    public ArbolBinario() {
        raiz = null;
    }

    public void insertar(int valor) {
        raiz = insertarRec(raiz, valor);
    }

    private Nodo insertarRec(Nodo raiz, int valor) {
        if (raiz == null) {
            return new Nodo(valor);
        }

        if (valor < raiz.valor) {
            raiz.izquierda = insertarRec(raiz.izquierda, valor);
        } else if (valor > raiz.valor) {
            raiz.derecha = insertarRec(raiz.derecha, valor);
        }

        return raiz;
    }

    public void imprimirPorNiveles() {
        if (raiz == null) {
            return;
        }

        Queue<Nodo> cola = new LinkedList<>();
        cola.add(raiz);

        while (!cola.isEmpty()) {
            int nivelSize = cola.size();

            for (int i = 0; i < nivelSize; i++) {
                Nodo nodo = cola.poll();
                System.out.print(nodo.valor + " ");

                if (nodo.izquierda != null) {
                    cola.add(nodo.izquierda);
                }

                if (nodo.derecha != null) {
                    cola.add(nodo.derecha);
                }
            }

            System.out.println();
        }
    }

    public static void main(String[] args) {
        ArbolBinario arbol = new ArbolBinario();

        arbol.insertar(9);
        arbol.insertar(5);
        arbol.insertar(17);
        arbol.insertar(12);
        arbol.insertar(8);
        arbol.insertar(3);
        arbol.insertar(19);
        arbol.insertar(10);
        arbol.insertar(11);
        arbol.insertar(6);


        System.out.println("*****************************************");
        System.out.println("Nasheli");
        System.out.println("");
        System.out.println("Estructura del árbol binario por niveles:");
        arbol.imprimirPorNiveles();
    }
}

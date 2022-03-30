#date: 2022-03-30T17:08:33Z
#url: https://api.github.com/gists/ed3891cc31ba531dbb9051e6f571f3a3
#owner: https://api.github.com/users/jaimeHMol

# -*- coding: utf-8 -*-
"""
    Autodesafio para construir un objeto que almacene arboles binarios, y a partir de este
    imprimir la información que se almacene en los nodos de estos arboles yendo de arriba
    hacia abajo y de izquierda a derecha.

    Usa Clases (Padre e hija) para el manejo del arbol binario
    Función recursiva para la impresión de los nodos segun la manera solicitada

    autor: jaimeHMol
    Abril 2018
"""


class nodo:
    """ Define la características (variables) y acciones (métodos) de un nodo
        de un arbol binario
    """
    ident = 0
    valor = " "
    hijoIzq = 0
    hijoDer = 0
    padre = 0

    def __init__(self, ident, valor, padre, hijoIzq, hijoDer):
        self.ident = ident
        self.valor = valor
        self.hijoIzq = hijoIzq
        self.hijoDer = hijoDer
        self.padre = padre

    def actualizaNodo(self, valor, padre, hijoIzq, hijoDer):
        self.valor = valor
        self.hijoIzq = hijoIzq
        self.hijoDer = hijoDer
        self.padre = padre


class arbolBin:
    nodos = []
    cantNodos = 0

    def __init__(self, nodoVal):
        self.cantNodos = 1
        self.nodos.append(nodo(self.cantNodos - 1, nodoVal, 1, 0,0))
        #self.nodos[self.cantNodos] = nodo(self.cantNodos, nodoVal, 1, 0,0)
        #return 'Creado nodo ' + str(self.cantNodos)


    def addHijoIzq(self, nodoPadre, nodoVal):
        self.cantNodos = self.cantNodos + 1

        self.nodos.append(nodo(self.cantNodos - 1, nodoVal, nodoPadre, 0,0))

        self.nodos[nodoPadre].actualizaNodo(self.nodos[nodoPadre].valor,
                  self.nodos[nodoPadre].padre,
                  self.cantNodos,
                  self.nodos[nodoPadre].hijoDer)

        return 'Creado nodo izquierdo ' + str(self.cantNodos) + ' con padre ' + str(self.nodos[nodoPadre].padre)


    def addHijoDer(self, nodoPadre, nodoVal):
        self.cantNodos = self.cantNodos + 1

        self.nodos.append(nodo(self.cantNodos - 1, nodoVal, nodoPadre, 0,0))

        self.nodos[nodoPadre].actualizaNodo(self.nodos[nodoPadre].valor,
                  self.nodos[nodoPadre].padre,
                  self.nodos[nodoPadre].hijoIzq,
                  self.cantNodos)

        return 'Creado nodo derecho ' + str(self.cantNodos) + ' con padre ' + str(self.nodos[nodoPadre].padre)


def printNodos (arbol, pos):
    """
    Funcion recursiva para la impresión del arbol según lo solicitado
    """
    if pos == 0:
        global busq
        busq = []
        global indice
        indice = -1

    if arbol.nodos[pos].hijoIzq != 0:
        busq.append(arbol.nodos[pos].hijoIzq)

    if arbol.nodos[pos].hijoDer != 0:
        busq.append(arbol.nodos[pos].hijoDer)

    print(arbol.nodos[pos].valor)

    indice = indice + 1
    if indice > len(busq) - 1:
       return
    else:
        printNodos(arbol, busq[indice] - 1)


# MAIN

# Crea el un arbol binario (definiendo su nodo raiz)
arbolTest = arbolBin("tst1")
arbolTest.cantNodos

# Agrega nodos "ramas" al arbol creado anteriormente
arbolTest.addHijoIzq(0,"tst2")
arbolTest.addHijoIzq(1,"tst3")
arbolTest.addHijoDer(0,"tst4")
arbolTest.addHijoDer(1,"tst5")
arbolTest.addHijoDer(3,"tst6")

arbolTest.cantNodos
arbolTest.nodos[2].padre

# Imprime el arbol recien creado
printNodos(arbolTest, 0)

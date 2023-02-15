#date: 2023-02-15T16:52:10Z
#url: https://api.github.com/gists/5676f42cbba3fed65e05659be3162e0b
#owner: https://api.github.com/users/Siliutors

import numpy as np
import pandas as pd


def suma_movil(vector, n):
    longitud = len(vector)
    suma = [sum(vector[i:i+n]) for i in range(longitud-n+1)]
    return suma

def deshacer_agupaciones(vector, n_veces):
    salida = list()
    for i in range(len(vector)):
        elemento_a_repetir = vector[i]
        salida = salida + list(np.repeat(elemento_a_repetir, n_veces))
    return salida

def cambios_monotonia(vector):
    longitud = len(vector)
    x = []
    flag = 0 #se activa con la primer min
    sentido = None
    for i in range(longitud-1):
   
        if sentido == None:
            x.append(0)
            
            if vector[i] > vector[i+1]:
                sentido = 'Decreciente'
            elif vector[i] < vector[i+1]:
                sentido = 'Creciente'
            else:
                continue
        elif sentido == 'Decreciente':
            if vector[i] > vector[i+1]:
                sentido = 'Decreciente'
                x.append(0)
            elif vector[i] < vector[i+1]:
                sentido = 'Creciente'
                x.append(-1)
            else:
                x.append(0)          
        else:
            if vector[i] > vector[i+1]:
                sentido = 'Decreciente'
                x.append(1)
            elif vector[i] < vector[i+1]:
                sentido = 'Creciente'
                x.append(0)
            else:
                x.append(0)             
    
            
        #Si no hay min. previo no se tiene en cuenta el max.
        if flag == 0 and x[i] == -1:
            flag = 1
        elif flag == 0 and x[i] == 1:
            x[i] = 0 #Cambia indicador de max. por 0
        else:
            pass
    x = x + [0]
    return x

def encontrar_ciclos(vector, imprimir_tabla = False):
    ciclos = []
    ciclo_actual = {}
    pos_inicial = None

    for i in range(len(vector)):
        valor_actual = vector[i]

        if pos_inicial is None:
            if valor_actual == -1:
                pos_inicial = i
                ciclo_actual['pos_ini'] = pos_inicial
                ciclo_actual['pos_fin'] = None
                ciclo_actual['valores'] = [valor_actual]
            else:
                continue

        else:
            ciclo_actual['valores'].append(valor_actual)

            if valor_actual == 1:
                    ciclo_actual['pos_fin'] = i
                    ciclos.append(ciclo_actual)
                    ciclo_actual = {}
                    pos_inicial = None
    

                
                
    if imprimir_tabla:            
        # Imprimir tabla de ciclos
        print('Número de ciclo | Posición de -1 | Posición de 1 | Valores')
        print('----------------------------------------------------------')
        for i, ciclo in enumerate(ciclos):
            print(f'{i+1:<16} {ciclo["pos_ini"]:<16} {ciclo["pos_fin"]:<16} {ciclo["valores"]}')
        
    # Crear DataFrame de ciclos
    df_ciclos = pd.DataFrame(ciclos, columns=['pos_ini', 'pos_fin', 'valores'])
    df_ciclos.index.name = 'num_ciclo'

    return df_ciclos


def optimize_prices(price_series = None, cycle_length=4, n_cycles=60000, n_years=2):
    n_hours = n_years * 365 * 24
    prices = np.random.normal(50, 40, n_hours) if price_series is None else price_series
    
    suma_movil_precios = suma_movil(prices, cycle_length)
    x = cambios_monotonia(suma_movil_precios)
    
    n_ciclos_x = sum([x for x in x if x == 1])
    
    df_ciclos = encontrar_ciclos(x)
    df_ciclos['precio_ini'] = [suma_movil_precios[i] for i in df_ciclos['pos_ini']]
    df_ciclos['precio_fin'] = [suma_movil_precios[i] for i in df_ciclos['pos_fin']]
    df_ciclos['spread'] = df_ciclos['precio_fin'] - df_ciclos['precio_ini']
    df_ciclos['rank_spread'] = df_ciclos['spread'].rank()
    
    if n_ciclos_x <= n_cycles:
        pass
        #x_optimo = x
        #max_beneficio = np.dot(x,prices) #sumaproducto
    else:
        #Optimizar solucion
        pass
        
    # df = {
    #         "price": prices,
    #         "x":  deshacer_agupaciones(x, cycle_length),
    #         "beneficio": max_beneficio
    #     }        
    
    
    return df_ciclos, suma_movil_precios, x

#Outputs
#df = optimize_prices()
#df = pd.DataFrame(df)
#nombre_fichero = "output_baterias.csv"
#df.to_csv(nombre_fichero, index=True, sep=';', decimal="," , header=True)

df_ciclos, suma_movil_precios, x = optimize_prices()

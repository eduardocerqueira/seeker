#date: 2023-06-21T16:47:42Z
#url: https://api.github.com/gists/1d3d524462736033c2166d6c65ad7830
#owner: https://api.github.com/users/n3cr0murl0c

def nui_validator(data):
    #data =input('numero de cedula: ')
    """
    Usada en combinacion con RegexValidator
    con regex = r"^\+?1?\d{10}$",
        message=("La Número Único de Identificación tiene solamente 10 digitos"),
    para evitar la inserción de datos diferentes.
    """
    if len(data)==10:
        if int(data[0:2])>=1 and int(data[0:2])<=24:#Valido si la region son las dos primros digitos
            ultimo_digito = data[-1]
            sum_pares = int(data[1]) + int(data[3]) + int(data[5]) + int(data[7])
            
            sum_impares=0
            for i in range(len(data)):
                if i==0 or i==2 or i==4 or i == 6 or i==8:
                    print(f"i:{i} data: {data[i]}")
                    print(f"lambda: {(lambda x: x*2 - 9 if x*2 > 9 else x*2)(int(data[i]))}")
                    sum_impares = sum_impares + (lambda x: x*2 - 9 if x*2 > 9 else x*2)(int(data[i]))
            print(f"suma de pares: {sum_pares}")
            print(f"suma de impares: {sum_impares}")
            suma_total = sum_pares+sum_impares
            print(f"Suma total: {suma_total}")
            #extraemos el primero digito
            primer_digito_suma = str(suma_total)[0]
            print(f" primer digito suma: {primer_digito_suma}")
            #Obtenemos la decena inmediata
            decena = (int(primer_digito_suma) + 1)  * 10
            print(f"decena: {decena}")
            #Obtenemos la resta de la decena inmediata - la suma_total esto nos da el digito validador
            #Si el digito validador es = a 10 toma el valor de 0
            digito_validador = 0 if (decena - suma_total==10) else decena - suma_total
            print(f"digito validador: {digito_validador}")
            print(f"ultimo digito: {ultimo_digito}")
                
            #Validamos que el digito validador sea igual al de la cedula
            if digito_validador != int(ultimo_digito):
                print('la cedula: ' + data + ' es incorrecta')
                # raise ValidationError('Esta cédula es incorrecta')
                raise ValueError
            print('la cedula:' + data + ' es correcta')
    else:
        print("cedula incompleta")
        raise ValueError         
            
        
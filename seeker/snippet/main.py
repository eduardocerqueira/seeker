#date: 2024-02-07T16:47:59Z
#url: https://api.github.com/gists/4912e53b72e4a79a0aa2e35306887f75
#owner: https://api.github.com/users/TinchoBus

def main():
    nombre =input('Por favor ingrese su nombre completo: ')
    apellido= input('Ingrese su apellido: ')
    telefono = int( input('Ingrese su número telefonico: '))
    email= input('Ingrese por favor su correo electronico: ')

    print('Hola', nombre, apellido + ', en breve recibirás un correo a: ', email + ',') 

if __name__ == '__main__':
    main()
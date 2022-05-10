#date: 2022-05-10T17:01:46Z
#url: https://api.github.com/gists/492c7abb86efa26864d7ba4876fa1217
#owner: https://api.github.com/users/fridelia

# poner un def, para acortar codigo..regresar a input de archivo, o menú general
from pathlib import Path

current_directory = Path.cwd()
path = Path(current_directory) 
#menú de opciones #listar, leer y editar

#LISTAR
for dir in path.iterdir():
    if dir.is_file() and dir.suffix == '.txt':
        print(dir.name)
        
#menu para retornar o elegir
#LEER
#añadir .txt a concatenar para eliminar la necesidad de escribirla


def validar(archivo1):
    while True:
        data1 = input(archivo1)
        if data1.is_file():
            print('menú')
        else:
            print('Nombre incorrecto, intenta de nuevo.')

archivo = validar('nombre del archivo: ')



#path_file = path / archivo
#if path_file.exists():
 #   with open(path_file, 'r') as file:
  #      content = file.read()
   #     print(content)

#if path_file.exists():
    #path_file = True

   # while True:
    #    if path_file.exists():
     #       with open(path_file, 'r') as file:
      #              content = file.read()
       #             print(content)
   # break

    #        af == True
#usar def para menu de opciones
#while eleccion
#if eleccion == x
#ingresar metodo fef
#uSAR ELIF
#eleccion = debe ser ingresar el numero para correr el menu




#EDITAR
#editar, menú de opciones, sobrescribir, añadir al final del archivo
#menú de leer y editar al mismo tiempo
#finalizar con booleano o palabra que detone cierre
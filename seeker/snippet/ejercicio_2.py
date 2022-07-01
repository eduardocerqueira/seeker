#date: 2022-07-01T17:01:08Z
#url: https://api.github.com/gists/95363c24701abab9d60b08d7a8896dc5
#owner: https://api.github.com/users/CSilvi

s='si'
while('si' in s):
    pay =input('Indique el total a pagar')
    y = int(pay)
    personas =input('cuantas personas se dividirá la cuenta')
    y_2 = int(personas)
    prop =input('porcentaje de propina a incluir')
    y_3 = int(prop)
    imp =input('un porcentaje de impuestos')
    y_4 = int(imp)

    if((y>0) and (y_2>0) and (y_3>0) and (y_4>0)):
          y = y + ((y*y_3)/100) + ((y*y_4)/100)
          x = y/y_2
          print('Total a pagar: ', y)
          print('Total a pagar por cada persona.: ', x)
    else:
        print('Ingrese valor valido: Los valores deben ser mayores a 0')

    s= input('Si desea realizar otra operacion escriba -si- en caso contrario -no-')
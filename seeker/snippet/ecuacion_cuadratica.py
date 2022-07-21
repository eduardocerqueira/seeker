#date: 2022-07-21T16:56:27Z
#url: https://api.github.com/gists/9fffc2cf65606e732c9da9a06de7ae36
#owner: https://api.github.com/users/abrahamjese

def formula_general(a,b,c):

    discriminante= b**2-4*a*c
    d=abs(discriminante)

    if discriminante>0 or discriminante==0:
        x1=(-b-d**0.5)/2*a
        x2=(-b+d**0.5)/2*a
        print('Las raices son: x1= '+str(round(x1,2))+' & x2= '+str(round(x2,2))+'.')

    if discriminante<0:
        x= (-b/2*a,d/2*a)
        print('Las raices son: ')
        print('x1= '+str(x[0])+'+'+str(x[1])+'i')
        print('x2= '+str(x[0])+'-'+str(x[1])+'i')


def run():
    print("""
        PROGRAMA PARA RAICES DE ECUACION CUADRATICA...
    """)
    a= float(input('Cual es el coeficiente cuadratico: '))
    b= float(input('Cual es el coeficiente lineal: '))
    c= float(input('Cual es el coeficiente independiente: '))
    formula_general(a,b,c)

if __name__=='__main__':
    run()
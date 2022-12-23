#date: 2022-12-23T16:57:16Z
#url: https://api.github.com/gists/71bc8713e6428f73ea70eafa949cf1ce
#owner: https://api.github.com/users/luisrco

#!/usr/bin/python
# -*- coding: utf-8 -*-
# www.pythondiario.com

from tkinter import Tk, Text, Button,Label, END,ttk, re, Entry,StringVar,Frame
from tkinter import *
import math



def salir():
  salir = app.destroy()

def hacer_click():
 try:
  _valor = int(entrada_texto.get())
  opcion = _valor
  if opcion is 1:
      fi = float(math.radians(26.6))
      pe = float(1400)
      c = float(0)
      nq = float(11)
      nc = float(25)
      ng = float(9)
      qmax = round(float(1.3 * c * nc + (pe * nq + 0.4 * pe * ng) * 1E-04))
      qu = round(qmax / 3)
      etiqueta2.config(text=qu)
      etiqueta3.config(text=pe)
      etiqueta.config(text=_valor)
  elif opcion is 2:
      fi = float(math.radians(31.5))
      pe = float(1500)
      c = float(0)
      nq = float(21)
      nc = float(33)
      ng = float(19)
      qmax = round(float(1.3 * c * nc + (pe * nq + 0.4 * pe * ng) * 1E-04))
      qu = round(qmax / 3)
      etiqueta2.config(text=qu)
      etiqueta3.config(text=pe)
      etiqueta.config(text=_valor)
  elif opcion is 3:
      fi = float(math.radians(36.5))
      pe = float(1700)
      c = float(0)
      nq = float(18)
      nc = float(54)
      ng = float(50)
      qmax = round(float(1.3 * c * nc + (pe * nq + 0.4 * pe * ng) * 1E-04))
      qu = round(qmax / 3)
      etiqueta2.config(text=qu)
      etiqueta3.config(text=pe)
      etiqueta.config(text=_valor)
  elif opcion is 4:
      fi = float(math.radians(32.7))
      pe = float(1750)
      c = float(0.43)
      nq = float(23)
      nc = float(34)
      ng = float(21)
      qmax = round(float(1.3 * c * nc + (pe * nq + 0.4 * pe * ng) * 1E-04))
      qu = round(qmax / 3)
      etiqueta2.config(text=qu)
      etiqueta3.config(text=pe)
      etiqueta.config(text=_valor)
  elif opcion is 5:
      fi = float(math.radians(25))
      pe = float(1600)
      c = float(0.14)
      nq = float(10)
      nc = float(30)
      ng = float(8)
      qmax = round(float(1.3 * c * nc + (pe * nq + 0.4 * pe * ng) * 1E-04))
      qu = round(qmax / 3)
      etiqueta2.config(text=qu)
      etiqueta3.config(text=pe)
      etiqueta.config(text=_valor)
  elif opcion is 6:
      fi = float(math.radians(30))
      pe = float(1750)
      c = float(0.37)
      nq = float(19)
      nc = float(30)
      ng = float(19)
      qmax = round(float(1.3 * c * nc + (pe * nq + 0.4 * pe * ng) * 1E-04))
      qu = round(qmax / 3)
      etiqueta2.config(text=qu)
      etiqueta3.config(text=pe)
      etiqueta.config(text=_valor)
  else:
      print("ninguna opcion valida")

 except ValueError:
  etiqueta.config(text="Introduce un numero!")

def funcion():
 global acero
 acero=5.08
 try:
  opcion = 1
  if opcion is 1:
     P = float(input("ingresa la carga axial de la fundacion, en kg="))
     qadm = float(input("ingrese el valor del esfuerzo del terreno sugerido con anterioridad, en kg/cm2="))
     pe = float(input("ingrese el valor del peso especifico del terreno sugerido con anterioridad, en kg/m3="))
     fc = 210
     fy = 4200
     d = 30
     r = 7.5
     hf = d + r
     prof = float(input("ingresa la profundidad de fundacion, en metros="))
     niu = 1.175
     hp = prof * 100 - hf
     Areq = niu * P / qadm
     Bx = float(round(math.sqrt(Areq)))
     By = Bx
     Area = By * Bx
     Pr=1.07*P
     Pu=1.4*Pr
     sigma=Pu/Area
     a= float(input("ingresa el ancho de la columna sentido x, en centimetros="))
     b = float(input("ingresa el ancho de la columna sentido y, en centimetros="))
     A1=a*b
     n=0.5*(Bx-a)
     c=n-d
     Mu=sigma*Bx*n/2
     d1=math.sqrt(Mu/(0.1448*fc*Bx))
     if d1>d:
         d=d1
     else:
         d = d

     Vu=sigma*Bx*c
     vu=Vu/(0.85*Bx*d)
     vc=0.53*math.sqrt(fc)
     if vu>vc:
         d=Vu/(0.85*Bx*vu)
     else:
         d = d

     Vup = Pu-sigma *math.pow((a+d),2)
     bo=4*(a+d)
     vup = Vup / (0.85 * bo * d)
     vcp = 1.06 * math.sqrt(fc)
     if vup > vcp:
         d = Vu / (0.85 * bo * vu)
     else:
         d = d

     if Mu/(0.8044*fy*d)>=0.002*Bx*hf:
      k=Mu/(0.8044*fy*d)
     else:
         k = 0.002*Bx*hf

     Pc=0.59*fc*A1
     Pca=k*Pc
     if Pc >= Pu and Pca >=2*Pu \
             :
      d=d
     else:
      d = d+0.05

      kas= Mu/(0.8044*fy*d)
      Asmin=0.002*Bx*hf
      if kas>= Asmin:
        kas=kas
      else:
        kas= kas
        print("acero", kas)

     etiqueta7.config(text="Bx")
     etiqueta8.config(text="By")
     etiqueta9.config(text="hf")
     etiqueta10.config(text="Asx")
     etiqueta11.config(text=Bx)
     etiqueta12.config(text=By)
     etiqueta13.config(text=hf)
     etiqueta14.config(text=k)
     etiqueta15.config(text="responsable")
 except ValueError:
        etiqueta.config(text="Introduce un numero!")



def calcular():
 print("esta pasando por aqui",2*6)

app = Tk()
app.title("Aprendiendo a trabajar con App Grafica")



etiqueta = Label(app, text="Valor")
etiqueta.grid(column=1, row=2, sticky=(W,E))
etiqueta2 = Label(app, text="valor de q")
etiqueta2.grid(column=2, row=2, sticky=(W,E))
etiqueta3 = Label(app, text="valor de pe")
etiqueta3.grid(column=3, row=2, sticky=(W,E))
etiqueta4 = Label(app, text="opcion")
etiqueta4.grid(column=1, row=3, sticky=(W,E))
etiqueta5 = Label(app, text="valor de q")
etiqueta5.grid(column=2, row=3, sticky=(W,E))
etiqueta6 = Label(app, text="valor de pe")
etiqueta6.grid(column=3, row=3, sticky=(W,E))
etiqueta7 = Label(app, text="Bx")
etiqueta7.grid(column=1, row=8, sticky=(W,E))
etiqueta8 = Label(app, text="By")
etiqueta8.grid(column=2, row=8, sticky=(W,E))
etiqueta9 = Label(app, text="hf")
etiqueta9.grid(column=3, row=8, sticky=(W,E))
etiqueta10 = Label(app, text="As")
etiqueta10.grid(column=1, row=10, sticky=(W,E))
etiqueta11 = Label(app, text="Bx")
etiqueta11.grid(column=1, row=9, sticky=(W,E))
etiqueta12 = Label(app, text="By")
etiqueta12.grid(column=2, row=9, sticky=(W,E))
etiqueta13 = Label(app, text="hf")
etiqueta13.grid(column=3, row=9, sticky=(W,E))
etiqueta14 = Label(app, text="Asx")
etiqueta14.grid(column=2, row=10, sticky=(W,E))
etiqueta15 = Label(app, text="Responsable")
etiqueta15.grid(column=2, row=11, sticky=(W,E))



boton = Button(app, text="ingrese opcion!", command=hacer_click)
boton.grid(column=1, row=1)
boton1 = Button(app, text="datos fundacion!", command=funcion)
boton1.grid(column=1, row=7)
boton2 = Button(app, text="salir!", command=salir)
boton2.grid(column=1, row=12)


valor = ""
opcion = ""
entrada_texto = Entry(app, width=30, textvariable=opcion)
entrada_texto.grid(column=2, row=1)



app.mainloop()










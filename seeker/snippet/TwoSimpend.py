#date: 2026-03-12T17:40:00Z
#url: https://api.github.com/gists/40a95fe056ccfac3dd559020991b36b8
#owner: https://api.github.com/users/nankasitrahana-eng

import numpy as np 
import math
import tkinter as tk
fenetre = tk.Tk()
fenetre.geometry("300x500")
fenetre.title("fentre double pendule")
def fonc1():
  import matplotlib.pyplot as plt
  from matplotlib.animation import FuncAnimation
  tmax = 30    
  l1 =float(i1.get())
  l2 = float(i2.get())
  g = 9.81
  m1 = float(i3.get())
  m2 = float(i4.get())
  dt = 0.01

  t = np.arange(0, tmax, dt)

  t1 = np.zeros(len(t))
  t2 = np.zeros(len(t))
  w1 = np.zeros(len(t))
  w2 = np.zeros(len(t))

  w1i =f3.get()
  w2i= f4.get()
  t1i= g1.get()
  t2i = g2.get()
  w1[0]=float(w1i)
  w2[0]=float(w2i)
  t1[0]=(math.pi*float(t1i)/180)
  t2[0]=(math.pi*float(t2i)/180)
  def j1(t,t2,t1,w1,w2):
     dl=t2-t1
     A = (1/3 * m1 + m2) * l1**2
     B = 1/2 * m2 * l1 * l2 * np.cos(t1 - t2)
     C = 1/2 * m2 * l1 * l2 * w2**2 * np.sin(t1 - t2) + (1/2 * m1 + m2) * g * l1 * np.sin(t1)
    
     E = 1/2 * m2 * l1 * l2 * np.cos(t1 - t2)
     D = 1/3 * m2 * l2**2
     F = -1/2 * m2 * l1 * l2 * w1**2 * np.sin(t1 - t2) + 1/2 * m2 * g * l2 * np.sin(t2)
     a1=(B*F-C*D)/(A*D-B*E)
     return a1
 
  def j2(t,t2,t1,w1,w2):
    dl=t2-t1
    A = (1/3 * m1 + m2) * l1**2
    B = 1/2 * m2 * l1 * l2 * np.cos(t1 - t2)
    C = 1/2 * m2 * l1 * l2 * w2**2 * np.sin(t1 - t2) + (1/2 * m1 + m2) * g * l1 * np.sin(t1)
    
    E = 1/2 * m2 * l1 * l2 * np.cos(t1 - t2)
    D = 1/3 * m2 * l2**2
    F = -1/2 * m2 * l1 * l2 * w1**2 * np.sin(t1 - t2) + 1/2 * m2 * g * l2 * np.sin(t2)
    a2=(C*E-A*F)/(A*D-B*E)
    return a2
  def f1(t,w1):
      return w1
  def f2(t,w2):
      return w2
  for i in range(len(t)-1):
    # premier
      k1_t1 = f1(t[i], w1[i])
      k1_t2 = f2(t[i], w2[i])
      k1_w1 = j1(t[i], t2[i], t1[i], w1[i], w2[i])
      k1_w2 = j2(t[i], t2[i], t1[i], w1[i], w2[i])
    
    # deusieme
      k2_t1 = f1(t[i] + (dt / 2), w1[i] + (k1_w1 * dt / 2))
      k2_t2 = f2(t[i] + (dt / 2), w2[i] + (k1_w2 * dt / 2))
      k2_w1 = j1(t[i] + (dt / 2), t2[i] + (k1_t2 * dt) / 2, t1[i] + (k1_t1 * dt) / 2, w1[i] + (k1_w1 * dt) / 2, w2[i] + (k1_w2 * dt) / 2)
      k2_w2 = j2(t[i] + (dt / 2), t2[i] + (k1_t2 * dt) / 2, t1[i] + (k1_t1 * dt) / 2, w1[i] + (k1_w1 * dt) / 2, w2[i] + (k1_w2 * dt) / 2)
    
    # trois
      k3_t1 = f1(t[i] + (dt / 2), w1[i] + (k2_w1 * dt / 2))
      k3_t2 = f2(t[i] + (dt / 2), w2[i] + (k2_w2 * dt / 2))
      k3_w1 = j1(t[i] + (dt / 2), t2[i] + (k2_t2 * dt) / 2, t1[i] + (k2_t1 * dt) / 2, w1[i] + (k2_w1 * dt) / 2, w2[i] + (k2_w2 * dt) / 2)
      k3_w2 = j2(t[i] + (dt / 2), t2[i] + (k2_t2 * dt) / 2, t1[i] + (k2_t1 * dt) / 2, w1[i] + (k2_w1 * dt) / 2, w2[i] + (k2_w2 * dt) / 2)
    
    # quatre
      k4_t1 = f1(t[i] + dt, w1[i] + (k3_w1 * dt))
      k4_t2 = f2(t[i] + dt, w2[i] + (k3_w2 * dt))
      k4_w1 = j1(t[i] + dt, t2[i] + (k3_t2 * dt) , t1[i] + (k3_t1 * dt), w1[i] + (k3_w1 * dt), w2[i] + (k3_w2 * dt))
      k4_w2 = j2(t[i] + dt, t2[i] + (k3_t2 * dt), t1[i] + (k3_t1 * dt), w1[i] + (k3_w1 * dt), w2[i] + (k3_w2 * dt))
      t1[i+1]=t1[i]+dt*(k1_t1+2*k2_t1+2*k3_t1+k4_t1)/6
      t2[i+1]=t2[i]+dt*(k1_t2+2*k2_t2+2*k3_t2+k4_t2)/6
      w1[i+1]=w1[i]+dt*(k1_w1+2*k2_w1+2*k3_w1+k4_w1)/6
      w2[i+1]=w2[i]+dt*(k1_w2+2*k2_w2+2*k3_w2+k4_w2)/6

  x1=(l1)*np.sin(t1)
  y1=-(l1)*np.cos(t1)
  x2=(l1)*np.sin(t1)+(l2)*np.sin(t2)

  y2=-(l1)*np.cos(t1)-(l2)*np.cos(t2)

  fig1,ax1=plt.subplots()
  lin1,=ax1.plot([],[],'-',lw=6,        solid_capstyle='round')
  lin2,=ax1.plot([],[],'-',lw=6,solid_capstyle='round')
  lin3,=ax1.plot([],[],'--')
  lin4,=ax1.plot([],[],'--')
  lin5,=ax1.plot([],[],'--')
  lin6,=ax1.plot([],[],'--')  
  ax1.set_aspect('equal', adjustable='box') 
  ax1.set_xlim(-(l1+l2)-(1/3)*(l1+l2),(l1+l2)+(1/3)*(l1+l2))
  ax1.set_ylim(-(l1+l2)-(1/3)*(l1+l2),(l1+l2)+(1/3)*(l1+l2))
  ax1.axhline(0, color='black')  
  ax1.axvline(0, color='black') 
  ax1.grid()
  def uptdate(i):
    lin1.set_data([0,x1[i]],[0,y1[i]])
    lin2.set_data([x1[i],x2[i]],[y1[i],y2[i]])
    lin3.set_data([0,x1[i]],[y1[i],y1[i]])
    lin4.set_data([x1[i],x1[i]],[0,y1[i]])
    lin5.set_data([0,x2[i]],[y2[i],y2[i]])
    lin6.set_data([x2[i],x2[i]],[0,y2[i]])
    return lin1,lin2
  a=10
  
  ani = FuncAnimation(
    fig1,
    uptdate,
    frames=range(0,len(t),a),
    interval=dt*a*1000
)
  plt.show()


g1=tk.Entry (fenetre)
g2=tk.Entry (fenetre)
f3=tk.Entry (fenetre)
f4=tk.Entry (fenetre)
i1=tk.Entry (fenetre)
i2=tk.Entry (fenetre)
i3=tk.Entry (fenetre)
i4=tk.Entry (fenetre)
c1= tk.Label(fenetre, text="angle1:")
c2 = tk.Label(fenetre, text="angle2 :")
c3 = tk.Label(fenetre, text="vitesse angulaire1 :")
c4 = tk.Label(fenetre, text="vitesse angulaire2 :")
k1= tk.Label(fenetre, text="l1:")
k2= tk.Label(fenetre, text="l2 :")
k3= tk.Label(fenetre, text="m1 :")
k4= tk.Label(fenetre, text="m2 :")

k1.pack()
i1.pack()
k2.pack()
i2.pack()
k3.pack()
i3.pack()
k4.pack()
i4.pack()
c1.pack(pady=5)
g1.pack(pady=5)
c2.pack(pady=5)
g2.pack(pady=5)
c3.pack(pady=5)
f3.pack(pady=5)
c4.pack(pady=5)
f4.pack(pady=5)

bouton1 = tk.Button(fenetre, text="Valider",command=fonc1)
bouton1.pack(pady=5)




fenetre.mainloop()
#date: 2022-07-22T17:05:07Z
#url: https://api.github.com/gists/c3b0b5632fe0ac2556d4df81fb8ba2b9
#owner: https://api.github.com/users/EricSchraf

from math import *
from random import random,randint
from kandinsky import fill_rect,set_pixel,get_pixel

for n in range(8, 248):
  for m in range(320):
    X,YY=0,-.1
    Z=3
    U=(M-159.5)/160
    V=(N-127.5)/160
    W=1/sqrt(U*U+V*V+1)
    U*=W
    V*=W
    I=copysign(1, U)
    G=1
    E=X-I
    F=Y-I
    P=U*E+V*F-W*Z
    D=P*P-E*E-F*F-Z*Z+1
    if D>0: T=-P-sqrt(D)
    if T>0:X=X+T*U
    Y=Y+T*V
    Z=Z-T*W
    E=X-I
    F=Y-I
    G=Z
    P=2*(U*E+V*F-W*G)
    U=U-P*E
    V=V-P*F
    W=W+P*G
    I=-I
    GOTO50


FORN=8TO247:FORM=0TO319
X=0:Y=-.1:Z=3:U=(M-159.5)/160:V=(N-127.5)/160:W=1/SQR(U*U+V*V+1):U=U*W:V=V*W:I=SGNU:G=1
E=X-I:F=Y-I:P=U*E+V*F-W*Z:D=P*P-E*E-F*F-Z*Z+1:IFD>0T=-P-SQRD:IFT>0X=X+T*U:Y=Y+T*V:Z=Z-T*W:E=X-I:F=Y-I:G=Z:P=2*(U*E+V*F-W*G):U=U-P*E:V=V-P*F:W=W+P*G:I=-I:GOTO50
IFV<0P=2/V:V=-V*(.6*(INT(U*P)+INT(W*P)AND1)+.3)+.08
GCOL0,3-(48*V^.4+?(PAGE+5+M MOD4+N MOD4*4)-48)DIV16
PLOT69,4*M,4*N:NEXT,

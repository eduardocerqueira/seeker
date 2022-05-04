#date: 2022-05-04T17:09:18Z
#url: https://api.github.com/gists/1adf17cbd0cc19da9bfb06935e008935
#owner: https://api.github.com/users/elcio

c="AIZbYUN X>eE#D*5G*0'@CDE#DF+R-'_BTbM-*#M'F+*eE#?>-B-O*ee%RQ_bMZ#*NZT'*e*c+QcCF-DCdd=SR=_TMTb=UAB3T*#O0V'F-TX=F+0' MA-GR=_G=G6gFYU M+OAT'3J'3TO'+VBA<<'ccL*C_K8*# S=,QA0ABT=,ZEV*Ab&<W5!O_KQEVT'SaJ3=E=,3-*U9J'W+a'&W5d=O4#C#DFaN3S=O-M+O'S=FT=#'& I WC?4-*E#Sa79DEDcCO-TGCGCOPAYDO14*E#Ga7FG@@5d9f-CO+a<7d14E#CPNSCa3=E#OBC-Q*E@'S&' =dd_KWE*+<<BF8A3Yd+A0J6O8,P =?_K*E#F&<BF+J'6d'PN-G-TA7BA ?_*H=E#QNP7TGA=D*UPa0A-< j=!_*CWcCe-P&3-G1-Pa7F-&Z =12CWEU<70G%BF8D*TPa7 G-&bA1_DOD*>E#*INa XgGF5A xM+D+aN-'FYd2OEOP3N+cQFBJ< jBTBPNSC-Yd2DOBJ IA'3-G8MaBP'SFP7BS912o8M7fG%MaBaF-I9'P3-5?_dL-W@B-B3Tgc%Bj<-A'+S*U70'+cd!_LD*cVbF,SH%-bbA ISWE=N3Sc=D*cC_#* F 1C-B-bT51cCg+fb6D*5ABING>CG-G_#?CGF-TMN->E#DHQfWEOJI7b+cc"
r="~|,B'|>C*|<'a|?!*|!dD|&aA|@C+|%+F|9-=|8+-|7AN|6+=|5C=|4_R|3 B|2_Ld|1d*|0 -|Z b|Y'=|Xb-|W=*|V=-|U#=|T'-|S-+|RcH|Q=+|Paa|O*=|N '|MbB|LD=|Kccc|J'A|I' '|HccC|G++|F--|E##|D**|C==|B''|A  |_\n".split('|')
for x in r:c=''.join((i,i.upper()*2)[i.islower()]for i in c.replace(x[0],x[1:]))
print(c)
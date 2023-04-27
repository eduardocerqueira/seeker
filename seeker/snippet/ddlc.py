#date: 2023-04-27T16:45:57Z
#url: https://api.github.com/gists/c5308fb681ea37b63e3d8fd7f711b28b
#owner: https://api.github.com/users/Nat456Gaming

_F,_E,_D,_C,_B,_A="grey",(255,255,255),(0,0,0),'Score : ','white','[OK]'
club_c=[[132, 122, 120], [164, 148, 143], [196, 181, 173], [219, 206, 200], [101, 90, 96], [255, 255, 255], [116, 112, 92], [81, 64, 70], [57, 47, 57], [114, 90, 86], [180, 142, 117], [157, 110, 93], [128, 86, 74], [97, 64, 57], [212, 173, 145], [235, 206, 172]]
club="AB10C13D4C6BD2BEB2AB10C15D5C3DCAB3CB15C4D7C5BD2BABDEABC2B10C16B3DCABDFCB2A2B13C13B3CB2CF2CD3CBA2BC2B8CD3CD3C5BCBABDF2A2BCD4CBAB9C5D2C5BCABCDF2CDCBA2BCD3CBAB8C7B3A3BD2FDBC4DCB2AB2D2CBA2BAB4C4DC4BC2D3FE4AB3C3AC2B6C3D9C2D4F2G4AE7C3B5C4D7C3D4F2B2AG2C2GAB3CBAB2ABCD8CH3C2FD2F3BCDBGD2GBC2BCD2BE2C2D5F2DBI2ECDFD2F3C2BC2DFB2D2C2D2BE2C4D5CBI2JDCFD2F3DCGC2DC2BD2C2BKAE2H2EJA7HJADCFCD2F2G3AGAGAJAJ2BL2JAMNM3LNHJKLAEGJCKCBCDF2B3NJAKMN2JMNM3JM2L2BANL4AEA2O2KBCD2FLMJ4EN2M3AEN2ML3JHM3LKBE3ALOLKBCD2EN3ML2J2N2MLKM2N2JHN2M2NK3OPOBPKLMKPDCJMHEI3JN4J2MJHI2NLN2JO2KJOPOBO2K2MJB2JHI2NML3MN2I2HJEHI2N4LOBHJ2HJKLK2M2N5ML2KL4NH4EMJ2N4GJBHJ3HI3HM2N2I5H2NMH2IH2EJL3K3L2KOPOPO2KHJK2MN3I9JHIHJ2M3LK3L3KBKOKJ2H2L6I4HIH4IHJMJ2I3HJM2L2KOKLJ3E2L6I3H6N2JM5HI6BCP5OJEHNM2L2I3H3IHEJH6J2MKO2KL3O2KL3J2HI3H2IHIHI2HJ2M2J2NH3MKO3KLH2JLJ2NH3IHIH"
res,res_c,impr="A1250",[_E],""
#début GEMA
try:from os import *
except:print('err:Omega est requis')
from kandinsky import *;from random import *;from ion import *;from time import *
def p(a):print(a)
def fr(a,b,c,d,e):fill_rect(a,b,c,d,e)
def kd(a):return keydown(a)
def rch(a):return choice(a)
def ra(a,b):return randint(a,b)
def inp(a='',b=0):return input(a)
def re(a,b):rename(a,b)
def ds(a,b,c,d=(0,0,0),e=_B):draw_string(a,b,c,d,e)
def gm_er(a=""):pu("une erreur est survenue\n"+str(a));gm_r()
def gm_i(im,pal,t=4,po=0,py=0,suppr="",bs=80):
 r,i=0,0
 while r<len(im):
  s,n=im[r],'';r+=1
  while r<len(im)and'9'>=im[r]>='0':n+=im[r];r+=1
  nb=1 if n==''else int(n);c=pal[ord(s)-65]
  for j in range(nb):
    if str(s)!=str(suppr):fr(po+t*(i%bs),py+t*(i//bs),t,t,(c[0],c[1],c[2]))
    i+=1
def gm_tp(txt,ma=32,st=""):
 a,b,A,d=list(txt)+[''],'','\n'+st,[" ","\'",",","."];l=len(a)
 if l>ma:
  mb=int(ma)
  for j in range(l//ma):
   nl=True
   for i in range(1,ma//4):
    if d.count(a[mb-i])>=1:a[mb-i]=A;mb+=mb-i;nl=False;break
   if nl:a.insert(mb,A);mb+=mb
  for i in range(l):b+=a[i]
 else:b=str(txt)
 return b,l//ma
def gm_w(txt,perso="",rw=0):a=gm_tp(txt);ds(str(perso),0,180-a[1]*20);ds(str(a[0]),0,200-a[1]*20)
def gm_t(txt,perso="",img=0,pal=0,a=4,b=0,c=0,d=""):
 if img==0:gm_w(txt,perso)
 else:gm_i(img,pal,a,b,c,d);gm_w(txt,perso)
def gm_a(a,b,c,d,e,a5='#err:str?'):
 ds('[up]'+str(a),0,0)
 if e>=2:ds('[down]'+str(b),0,18)
 if e>=3:ds('[left]'+str(c),0,36)
 if e>=4:ds('[right]'+str(d),0,54)
 if e>=5:ds(_A+str(a5),0,72)
def gm_r():fr(0,0,320,225,_B)
def sl(a):sleep(a)
def gm_s():co=[(0,0,0),_B];fr(0,0,320,255,co[0]);fr(130,70,60,60,co[1]);fr(140,80,50,40,co[0]);fr(180,100,10,30,co[1]);fr(180,100,-20,10,co[1]);ds('eMa',192,112);ds('Engine',130,132);sl(2.8);fr(0,0,320,255,_F);sl(0.1);gm_r();ds(_A,0,0)
def tr(a=''):
 for i in range(32):a=a+rch(['_','-','(',')','@','O','D','{','}','#','%','&','=','[]','|',';','!','?','H','>','<','*',"'",'.','/'])
 return a
def tt(b,c,x=0,y=0):
 for i in b:c+=i;ds(c,x,y);sl(0.05)
def gm_c(a,b,c,d,e,a5=0):
 while True:
  if kd(1)and e>=1:return a
  if kd(2)and e>=2:return b
  if kd(0)and e>=3:return c
  if kd(3)and e>=4:return d
  if kd(4)and e>=5:return a5
def pu(a,c=_B,x=10,y=75,w=300,h=75):fr(x,y,w,h,(0,0,0));fr(x+1,y+1,w-2,h-2,c);ds(gm_tp(a,31," ")[0],x+2,y+2,_D,c);ds(_A,x+2,y+h-20,_D,c);gm_c(1,1,1,1,5,1);sl(0.5)
gm_s()#fin_GEMA_v0.9
def sv():open("ddlc_s.py","w").write("#sa,lo="+str(sa)+","+str(lo)+"#");open("ddlc_s.hpy","w").write("#sa,lo="+str(sa)+","+str(lo)+"#")
v,jna="","Jeu"
def su():gm_a(rch(["Continuer","Suivant"]),0,0,0,1);gm_c(1,1,1,1,5,1)
def sr(a,b):gm_w(a,b,1);sl(0.5);su()
def st(a,b,c,d,e="A"+impr):gm_t(a,b,c,d,4,0,0,e);su()
def po(a=[],d=[0,0,0,0]):
 for j in range(5):
  b=[];gm_r();sl(0.5)
  for i in range(5):b.append(rch(a+sayo_chr+nat_chr+yuri_chr))
  gm_a(b[0],b[1],b[2],b[3],5,b[4]);c=gm_c(b[0],b[1],b[2],b[3],5,b[4])
  if nat_chr.count(c)>=1:d[0]+=1
  elif sayo_chr.count(c)>=1:d[1]+=1
  elif yuri_chr.count(c)>=1:d[2]+=1
  else:d[3]+=1
 return d
try:
 from ddlc_s import *
 if open("ddlc_s.py","r").readline()!=open("ddlc_s.hpy","r").readline():p("Ce n\'est pas bien de tricher")
except:
 sa=[0,"#",0,"user","a"]
 try:from ddlc_natsuki import *;from ddlc_sayori import *;from ddlc_yuri import *;from ddlc_monika import *;sa,lo=[0,"Commencer",0,getlogin(),"a"],[0,0,0,0];sv()
 except:p("\nil manque des fichiers ddlc_\n")
pna=sa[3]
if sa[0]=="f":ae=inp("Ce fichier à été supprimé\nMerci de réinstaller ce script, ainsi que les personnages (ddlc_perso),et de supprimer ddlc_s")+Adieu
try:
 from ddlc_monika import *
 if sa[0]=="md":p("CESSE DE ME TORTURE AINSI, JE NE REVIENDRAIS PAS");mon,mon_c=0,0
except:
 if sa[4]=="m":
  mon,mon_c,mna=0,[_E],0
  if sa[4]!="md":
   try:re("ddlc_sayori.hpy","ddlc_sayori.py")
   except:gm_er()
   try:re("ddlc_yuri.hpy","ddlc_yuri.py");re("ddlc_natsuki.hpy","ddlc_natsuki.py")
   except:gm_er()
   inp("file ddlc_monika unreachable");p("Que se passe-t-il, non, ne me dis pas que tu m\'as supprimée ???")
   for i in range(3):sl(2);p(">>>restore(\'ddlc_monika\')\nan error occured\n"+rch(["non!","Aide-moi","nooooooon","ne me laisse pas",">_<","Pourquoi ???"]))
   p(">_< adieu");sl(0.5);sa=[0,"Commencer",0,sa[3],"md"];sv();pu("Redémarage")
 else:
   from ddlc_sayori import *
   try:
    for j in ("...","...","Q-Quoi...","...","Ceci...","Qu’est-ce que c\'est...?","Oh, non...","Non...","ça ne peut pas être ça","cela ne peut pas être tout ce qu’il y a","Qu’est-ce que c\'est ?","Qu’est-ce que je suis ?","arrêtez","S\'IL VOUS PLAIT ARRÊTEZ !"):gm_i(res,res_c);st(j,sna,sayo,sayo_c)
   except:gm_er()
try:from ddlc_yuri import *
except:yuri,yuri_c,yuri_chr,yna=0,[],[],v
try:from ddlc_natsuki import *
except:nat,nat_c,nat_chr,nna=0,[],[],v
try:from ddlc_sayori import *
except:
 if sa[4]=="a":pu("END")+END
 sayo,sayo_c,sayo_chr,sna=0,[],[],v
tt("ddlc.moe",jna+" original : http://");gm_w("Ce jeu est une adaptation non-officielle https://ddlc.moe/warning.html","Jeu original par TEAM SALVATO",0);sl(5);gm_r();fr(0,0,80,225,(255, 230, 244));ds("Doki Doki\nLitterature\nclub",0,80,_B,(255, 189, 225));gm_w("[down]aide\nv:a0.7",v);gm_t(v,v,nat,nat_c,3,110,0,"A");gm_t(v,v,yuri,yuri_c,3,30,0,"A");gm_t(v,v,sayo,sayo_c,3,50,50,"A");gm_t(v,v,mon,mon_c,3,175,50,"A");gm_a(sa[1],0,0,0,1)
if gm_c(1,2,1,1,5,1)==2:inp("https://rb.gy/otjfju")
if sa[1]=="Commencer":sa=[0,"Continuer",0,pna,sa[4]]
e=[(mna,mon,mon_c),(sna,sayo,sayo_c),(nna,nat,nat_c),(yna,yuri,yuri_c),(pna,sayo,sayo_c),(pna,mon,mon_c),(pna,nat,nat_c),(pna,yuri,yuri_c),(mna,monf,monf_c)]
def z(a):
 u=[("ça doit être difficile de lancer un nouveau club",5,"Oui, peu de personnes serait réellement motivée pour débuter une nouvelle chose",0,"Spécialement lorsqu\'il s\'agit d\'une chose qui n\'attire pas l'attention, comme la littérature",0,"Tu dois travailler dur, pour convaincre qu\'il s\'agit d'une activité fun",0),("qu\'est-ce que tu regardes ? Si tu veux dire qql chose,dit le!",2,"D-Désole...",6,nna+"...",3,"Si tu veux, tu peux juste l\'ignorer quand elle est de mauvaise humeur",1),("N\'oublies pas de sauvegarder, sinon, tu risques de perdre des données",0,"Ne t'inquiète pas, ce script sauvegarde automatiquement",0),(pna+" quel genre de livre lis-tu ?",3,"Hum,... mangas",7,"\"La tête de "+nna+" se lève soudainement\"",6,"\"On dirait qu\'elle veut dire qql chose, mais elle reste silencieuse\"",6),("Quel est ton genre littéraire favori ?",7,"Je dirais l\'horreur, c\'est un genre qui te transporte dans l\'histoire",3,"Huh, je déteste l'horreur",2,"C\'est vrai, tu as plutôt l\'habitude d\'écrire des choses mignonnes, non ?",0,"Q-Quoi ?",2,"Q-Qu\'est ce qui te fais penser ça ?",2,"Tu as laissé trainé un bout de papier au dernier rassemblement",0)]
 if sa[4]!="a":u[1]=("qu\'est-ce que tu regardes ? Si tu veux dire qql chose,dit le!",2,"D-Désole...",6,nna+"...",3,"",2)
 if sa[4]=="m":u=[("Tu savais que j'avais un compte twitter? C'est: lilmonix3"),("Je ne suis pas une pro en codage, j'espère ne pas avoir tout cassé"),("C'est triste que nous ne puissions atteindre le festival"),("Je voudrais pouvoir être avec toi, sans être bloquée dans ce jeu"),("Tu savais que ce jeu utilisait (https://discode.fr/code/gema)"),("Pardonne-moi si je me répète, ça doit être la joie/le stress")]
 for d in range(a):
  b=ra(0,len(u)-1)
  if sa[4]=="m":gm_i(club,club_c,8,0,0,"",40);st(u[b],e[8][0],e[8][1],e[8][2])
  else:
   for h in range(len(u[b])/2):st(u[b][h*2],e[u[b][h*2+1]][0],e[u[b][h*2+1]][1],e[u[b][h*2+1]][2]);gm_i(club,club_c,8,0,0,"",40)
def ch(ac):
 global sa;global lo;sa[0]=ac;sv();gm_r()
 if sa[4]=="a":phr=[("Heeeeeey!!!",v),("Je vis une fille, courant vers moi, agitant ses bras",v),("Cette fille est "+sna+", ma voisine et amie d\'enfance",v),("J\'ai encore oublié de me réveiller, mais je t\'ai rattrapé",1),("Comme tous les matins...",4),("Mais..., Sinon...",1),("Je remarque que "+sna+" tente de changer rapidement,de sujet",4),("Tu veux rejoindre un club ? ...Je savais que tu serais OK",1),("J\'avais pourtant dit non.Bon, on dirait que je n\'ai pas le choix",4),("Cette journée de cours, semblait ne plus finir",v),("Je repense a ce que "+sna+" m\'a dit...",v),("Club...",v),("Viens au club de littérature...",1),("Je n\'avais pas remarque que "+sna+" étais rentrée dans ma classe",4),("Et qu\'il ne restait aucun élève de ma classe",4),("Je decide donc de suivre Sayori vers la salle du club",4),("Eh, vous tous! Le nouvau membre est la",1),("Je te l\'ai deja dit, ne m\'appelle pas \'nouveau membre---\'",4),("Eh? J\'observe les alentours de la salle",v),("Bienvenue au club de littérature","fille 1",yuri,yuri_c),("Sérieusement "+sna+"? Tu as amené un garçon ?","fille 2",nat,nat_c),("Quel moyen de tuer l\'ambiance...","fille 2",nat,nat_c),("Ah "+pna+", quelle bonne surprise,\nbienvenue au club",0),("Les mots m\'echape dans de telle circonstance",v),("Ce club...",v),("...est plein de filles incroyablement mignonnes",v),("Voici "+nna+",toujours pleine d\'énergie",sna,nat,nat_c),("Et elle, c\'est "+yna+",la personne la plus intelligente de ce club!",sna,yuri,yuri_c),("Ne dit pas de choses pareilles",yna+", rougisante",yuri,yuri_c),(yna+",qui apparait plus timide et mature, semble avoir",v,yuri,yuri_c),("des difficultés à suivre des personnes comme"+sna+" et "+nna,v,yuri,yuri_c),("Ah...C\'est bien de vous rencontrer toutes les deux",pna),("Et il semblerait que tu connaisses déjà "+mna+", la présidente du club?",sna,mon,mon_c),("C\'est super de te revoir",0),(mna+" me souris gentillement",v,mon,mon_c),("Nous étions dans la même classe, mais avons rarement parlé",v,mon,mon_c),(mna+" était la fille la plus populaire de la classe, intelligente, magnifique...",v,mon,mon_c),("En bref, hors de ma porté",v,mon,mon_c),("Donc, la voir me sourire me semble vraiment un peu...",v,mon,mon_c),("T-Toi aussi, "+mna,5),("Il reste une place à côté de moi, ou à côté de "+mna+", assieds-toi où tu le souhaites",1),("Je décide de m\'assoir à côté de "+sna+", la seule personne que je connaisse ici",v,sayo,sayo_c),("Au fait, qu\'est-ce qui t\'as fait rejoindre notre club, "+pna+" ?",0),("Je redoutais cette question",v,mon,mon_c),("Quelque chose me dit que je ne devrais pas dire à "+mna,v,mon,mon_c),("que j’ai été pratiquement traîné ici par "+sna,v,mon,mon_c),("Alors..., je n\'avais pas encore rejoint de club",5),("et "+sna+" semble heureuse de me voir ici, donc...",5),("c’est bon, ne soit pas embarrassé",0),("on fera en sorte que tu te sentes comme chez toi, ok ?",0),int(10),("Vous rentrez, comme d'habitude avec "+sna,jna,sayo,sayo_c),("j'ai peur, peur de t'aimer plus que tu ne pourrais m'aimer",1),("je t'aime tellement que je pourrais en mourir",1),("Me considere ...",1),(("On est amis","Je t'aime "+sna,0,0,2),(1,1,0,0,2)),("Aujourd\'hui débute la journee du festival",jna),("Prenant le maximum de fournitures pour le festival",jna),("Vous décidez d'aller demander de l'aide a "+sna+" pour porter les fournitures",jna),("Vous vous rendez directement devant sa maison",jna),("Vous attendez longtemps "+sna+" devant sa porte",jna),("Connaissant "+sna+", elle doit dormir comme une marmotte",jna),("Vous décidez donc d'aller au club sans son aide",jna),("Une fois arrive au club (avec difficultés)",jna),("Vous remarquez que seul "+mna+" est présente",jna,mon,mon_c),("Et "+sna+",tu l'as laisse pendante...",0),("Comment sait-elle pour notre relation avec "+sna,v,mon,mon_c),("Je sais bien plus de choses que tu ne le sais...",0),("Est ce que tu pourrais tenir les flyers...",0),("Vous prenez les flyers, tout en essayant de les lire",v),("Vous remarquez la page ou se trouvent les poèmes",v),("Votre regard se porte directement sur le poème de "+sna,v),("SORS",("SORS DE MA TETE "*2+"\n")*15),("Vous décidez rapidement d'aller la chercher",jna),("Vous vous rendez directement devant sa maison",jna),("Vous attendez longtemps "+sna+" devant sa porte",jna),("Connaissant "+sna+", elle doit dormir comme une marmotte",jna),(sna+", réveille-toi, tu vas être en retard",jna),("N'obtenant pas de réponse à la dixième tentative,vous décidez d'entrer",jna),("Vous êtes devant la porte de sa chambre.",jna),("Échouant vos tentatives de réveiller "+sna,jna),("Vous ouvrez délicatement la porte de sa chambre...","Jeu"),("Sayo...",v,sayos,sayos_c)]
 if sa[4]=="b":phr=[("Nouvelle sauvegarde","GeMa - Save not found"),("Hey, est-ce que tu veux rejoindre un club ?",0),(mna+" était la fille la plus populaire de la classe, intelligente, magnifique...",v,mon,mon_c),("Donc, la voir me sourire me semble vraiment un peu...",v,mon,mon_c),("P-Pourquoi pas",5),("Merci, tu seras notre quatrième membre",0),("Bienvenue au club de littérature","fille 1",yuri,yuri_c),("Sérieusement "+mna+"? Tu as amené un garçon ?","fille 2",nat,nat_c),("Bien que tu soit la présidente du club, préviends-nous avant d'inviter un nouveau membre","fille 2",nat,nat_c),("Voici "+yna+" et "+nna+",les deux autres membres du club",0),("Ah...C\'est bien de vous rencontrer toutes les deux",pna),("Je décide de m\'assoir à côté de "+mna+", la seule personne que je connaisse ici",v,mon,mon_c),(mna+" me souris gentillement",v,mon,mon_c),int(10),("Ta présence ne me procure que le plaisir d\'une lame",3),("Depuis que chaque jour, je peux te voir, mon esprit se trouble",3),("Comme ce doux couteau, tu me remplies la tête",3),("M\'acceptes-tu",3),(("On est amis","Je t'aime "+yna,0,0,2),(1,1,0,0,2)),("Hahahaha",3)*5,("Yur...",7),(v,v,natv,natv_c),("Pourquoi "+nna+" viens de sortir en courant ?",0),("Oh, je comprends...",0),("Bon, il ne me reste qu'une chose a faire",0)]
 if sa[4]=="m":phr=[int(50)]
 if sa[4]=="md":phr=[("Rejoins le club de litterature. Ce jeu n'est pas fini",sna+", ami d'enfance",sayo,sayo_c),("Ne t'inquiete pas, nous serons seuls","JUST "+sna,sayo,sayo_c),("Il reste la fin, il ne reste que moi !","JUST "+sna,sayo,sayo_c),("Profitons en ensemble. Juste nous, sans ces autres scripts.","JUST "+sna,sayo,sayo_c)]
 f=[res,res_c,club,club_c]
 for ja in range(0+ac,len(phr)):
  d=2 if 74>ja>8 else 0
  if sa[4]=="b":d=2
  gm_i(f[0+d],f[1+d],8,0,0,"",40);sa=[ja,sa[1],sa[2],sa[3],sa[4]];sv()
  if ja==int(len(phr)) and sa[4]=="b":fr(0,0,300,50,_F);tt("os.remove('natsuki.chr')",">>>");ds("natsuki.chr removed",0,20);sl(2);fr(0,0,300,50,_F);tt("os.remove('yuri.chr')",">>>");ds("yuri.chr removed",0,20);re("ddlc_natsuki.py","ddlc_natsuki.hpy");re("ddlc_yuri.py","ddlc_yuri.hpy");sa=[0,mna,0,pna,"m"];sv();pu("Redémarage");p(ch(sa[0]))
  if ja==int(len(phr)) and sa[4]=="md":pu("NOOOON!!!");st("Je ne laisserais pas l'histoire se repeter, pas une fois de plus","M?n?k?",sayo,sayo_c);fr(0,0,300,50,_F);tt("os.delete('DDLC.py')",">>>");sa=["f",v,0,0,"f"];sv();p("Monika - Avec ce geste, plus personne ne pourra te...,Adieu\nMonika - Merci d'avoir joué")
  try:
   if phr[ja][0]=="SORS":ds(phr[ja][1],0,0),gm_c(1,1,1,1,5,1)
   if phr[ja][0]=="Sayo...":sa=[0,"Commencer",0,pna,"b"];sv();gm_i(sayos,sayos_c);re("ddlc_sayori.py","ddlc_sayori.hpy");return "GeMa - Une erreur fatal est survenue"
   if phr[ja][0]=="Yur...":
    for i in range(25):gm_t(tr()+"\n"+tr(),yna,yurid,yurid_c)
   gm_a(phr[ja][0][0],phr[ja][0][1],phr[ja][0][2],phr[ja][0][3],phr[ja][1][4]);gm_c(phr[ja][1][0],phr[ja][1][1],phr[ja][1][2],phr[ja][1][3],phr[ja][1][4])
  except:
   try:int(phr[ja][1]);a=int(phr[ja][1]);st(phr[ja][0],e[a][0],e[a][1],e[a][2])
   except:
    try:st(phr[ja][0],phr[ja][1],phr[ja][2],phr[ja][3])
    except:
     try:sr(phr[ja][0],phr[ja][1])
     except:
      try:z(phr[ja])
      except:p(phr[ja])
if sa[4]=="m":
 if sa[2]==1:st(rch(["Te revoir me redonne le sourire","Combien de temps était tu partie,\nje suis heureuse de te revoir","Je suis heureuse de te revoir"]),mna,monf,monf_c)
 if mna!="Monika":st("Oh, tu m'as donne un surnom :3",mna,monf,monf_c)
try:p(ch(sa[0]))
except:
 sa[2]=1;sv()
 if sa[4]=="m" or sa[4]=="b":p("Reviens vite")
 gm_er()# This comment was added automatically to allow this file to save.
# You'll be able to remove it after adding text to the file.

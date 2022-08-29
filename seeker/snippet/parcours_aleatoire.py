#date: 2022-08-29T16:57:32Z
#url: https://api.github.com/gists/ab6b55daf62ff31de101e859688b6c34
#owner: https://api.github.com/users/bebertii

from random import randint
import time


def val(t):
  t%=180
  if t<120 :
    return int(t*(120-t)/14.1)
    
  else:
    return 0
    

def affiche(taille_tuile=1,width=1000,height=800,lenteur=10000):
    import pygame
    max_len_pile=0
    
    successes, failures = pygame.init()
    print(f"{successes} successes and {failures} failures")
    

    screen = pygame.display.set_mode((width, height))
    screen.fill("#000000")
    pile=[(randint(0,width//taille_tuile),randint(0,height//taille_tuile))]
    couleur=1
    while len(pile)>0:
        
        pygame.event.get()
        pos=pile[-1]
        R=val(couleur/lenteur)
        V=val(couleur/lenteur+60)
        B=val(couleur/lenteur+120)
        if tuple(screen.get_at((pos[0]*taille_tuile,pos[1]*taille_tuile)))[:-1]==(0,0,0):
            pygame.draw.rect(screen, (R,V,B) , (pos[0]*taille_tuile,pos[1]*taille_tuile,taille_tuile,taille_tuile))
            couleur+=1
        #possibles=[1,3,5,7]
        possibles=[0,1,2,3,5,6,7,8]
        while len(possibles)>0:
            
            test=possibles.pop(randint(0,len(possibles)-1))
            #print()
            if 0<test%3-1+pos[0]<width//taille_tuile and 0<test//3-1+pos[1]<height//taille_tuile:
                if tuple(screen.get_at(((test%3-1+pos[0])*taille_tuile,(test//3-1+pos[1])*taille_tuile)))[:-1]==(0,0,0):
                    pile.append((test%3-1+pos[0],test//3-1+pos[1]))
                    if len(pile)>max_len_pile:
                        max_len_pile=len(pile)
                    break
        if not len(possibles):
            pile.pop(-1)
        
        #time.sleep(0.01)
        #pygame.display.update()
        #print(couleur)
        if couleur%1000==0:
            pygame.display.update()
    return max_len_pile


print(affiche(1,1000,1000,100))
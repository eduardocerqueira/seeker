#date: 2023-04-07T16:55:48Z
#url: https://api.github.com/gists/0a80d5505110c17dcf44f7320a1f5af0
#owner: https://api.github.com/users/Alexnerotd

import pygame
import sys

pygame.init()

size = (800, 500)
screen = pygame.display.set_mode(size)

pygame.display.set_caption("Escondidas")

# Cargar imagen y redimensionarla
# BANOS
bano_alex = pygame.image.load("Bano de Alex.jfif")
bano_alex = pygame.transform.scale(bano_alex, size)

bano_principal = pygame.image.load("Bano principal.jfif")
bano_principal = pygame.transform.scale(bano_principal, size)

bano_padres = pygame.image.load("Bano de madrastra y padrastro.jfif")
bano_padres = pygame.transform.scale(bano_padres, size)

# CUARTOS
mi_cuarto = pygame.image.load('Mi cuarto.jfif')
mi_cuarto = pygame.transform.scale(mi_cuarto, size)

cuarto_padres = pygame.image.load('Cuarto de madrastra.jfif')
cuarto_padres = pygame.transform.scale(cuarto_padres, size)

cuarto_h1 = pygame.image.load('Cuarto de hermana 1.jfif')
cuarto_h1 = pygame.transform.scale(cuarto_h1, size)

cuarto_h2 = pygame.image.load('Cuarto de hermana 2.jfif')
cuarto_h2 = pygame.transform.scale(cuarto_h2, size)

# Alternar entre las dos imágenes con la barra espaciadora
current_image = 0
images = [bano_alex, bano_principal, bano_padres, cuarto_padres, mi_cuarto, cuarto_h1, cuarto_h2]
titles = ['Bano de Alex', 'Bano principal', 'Bano de madrastra y padrastro', 'Cuarto de madrastra', 'Mi cuarto',
          'Cuarto de hermana 1', 'Cuarto de hermana 2']

font = pygame.font.SysFont('Arial', 30)

# Agregar el boton de salida
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                current_image = (current_image + 1) % len(images)

    # Dibujar la imagen actual en la pantalla
    screen.blit(images[current_image], (0, 0))

    # Dibujar el título de la imagen actual en la pantalla
    title = font.render(titles[current_image], True, (255, 255, 255))
    screen.blit(title, (size[0] // 2 - title.get_width() // 2, 20))

    pygame.display.update()

#date: 2023-01-17T16:50:23Z
#url: https://api.github.com/gists/6b54331b9844ad6a2640e5da7a68be57
#owner: https://api.github.com/users/horstjens

import pygame
import random


class Player:
	number = 1
	images = []
	
	def __init__(self, pos=None, color="#222222"):
		self.number = Player.number
		Player.number += 1
		if pos is None:
			self.pos = [Game.WIDTH/2, Game.HEIGHT/2]
		else:
			self.pos = pos
		self.color = color
		#self.image = pygame.Surface((150,100))
		#pygame.draw.rect(self.image, "#010101", (0,50,150,50))
		#pygame.draw.rect(self.image, "#222222", (50,0,50,50))
		#self.image.set_colorkey((0,0,0))
		#self.image.convert_alpha()
		self.image = Player.images[0] # car heading right
		#self.right = True
		self.rect = self.image.get_rect()
		self.rect.center = self.pos
		self.dx = 2
		self.dy = 7
	
	def update(self, delta_time):
		if self.dx < 0:
			self.image = Player.images[1]
		if self.dx > 0:
			self.image = Player.images[0]
		self.pos[0] += self.dx * delta_time/1000
		self.pos[1] += self.dy * delta_time/1000
			
		
		
	

class Game:

	WIDTH = 800
	HEIGHT = 600
	FPS = 120

	def __init__(self):
		# Initialize Pygame.
		pygame.init()
		self.font = pygame.font.SysFont(None, 32)

		# Set size of pygame window.
		self.screen=pygame.display.set_mode((Game.WIDTH,Game.HEIGHT))
		# Create empty pygame surface.
		self.background = pygame.Surface(self.screen.get_size())
		# Fill the background white color.
		self.background.fill(("#BBBBFF"))
		self.clock = pygame.time.Clock()
		self.load_resources()
		self.player1 = Player()
		self.playergroup = []
		self.playergroup.append(self.player1)
		
	def load_resources(self):
		img = pygame.image.load("taxi.png")
		img.convert_alpha()
		Player.images.append(img)
		img2 = pygame.transform.flip(img, True, False)
		img2.convert_alpha()
		Player.images.append(img2)
		
	
	def update(self, delta_time):
		# playergroup
		for p in self.playergroup:
			p.update(delta_time)
			
	def draw(self):
		# playergroup
		for p in self.playergroup:
			p.rect.center = p.pos
			self.screen.blit(p.image, p.pos)
	
	def run(self):
		running = True
		playtime = 0 # seconds
		while running:
			milliseconds = self.clock.tick(Game.FPS) 
			playtime += milliseconds/1000 # in seconds
			
			mytext1 = self.font.render(f"FPS: {self.clock.get_fps():.1f}", True, "#00FF00")
			mytext2 = self.font.render(f"playtime: {playtime:.1f} seconds", True, "#00FF00")
			
			
			for event in pygame.event.get():
				# User presses QUIT-button.
				if event.type == pygame.QUIT:
					running = False 
				elif event.type == pygame.KEYDOWN:
					# User presses ESCAPE-Key
					if event.key == pygame.K_ESCAPE:
						running = False
				   
			pressed_keys = pygame.key.get_pressed()
			
			if pressed_keys[pygame.K_w]:
				self.player1.dy -= 1
			if pressed_keys[pygame.K_s]:
				self.player1.dy += 1
			if pressed_keys[pygame.K_a]:
				self.player1.dx -= 1		
			if pressed_keys[pygame.K_d]:
				self.player1.dx += 1
			
				
			
			self.screen.blit(self.background, (0, 0))
			self.update(milliseconds)
			self.draw()
			self.screen.blit(mytext1, (10,10))
			self.screen.blit(mytext2, (200,10))
			
			
			pygame.display.flip()

		pygame.quit()
    
    
if __name__ == "__main__":
	mygame = Game()
	mygame.run()    
    
    
    
    
    
    
    
    
    
    
    
    
    



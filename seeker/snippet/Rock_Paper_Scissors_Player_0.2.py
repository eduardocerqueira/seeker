#date: 2022-09-23T17:25:13Z
#url: https://api.github.com/gists/df4099e54667e90f6c14cff44269640a
#owner: https://api.github.com/users/tictackode

class Player():
	name="Unknown"
	choice="none"
	victories=0
	losses=0
	def choiceRPS(self):
		options=["Rock","Paper","Scissors"]
		from random import randint
		self.choice=options[randint(0,2)]
	def returnChoice(self):
		return self.choice
	def setName(self,name):
		self.name=name
	def victory(self):
		self.victories+=1
	def loss(self):
		self.losses+=1
	def showStats(self):
		print(self.name+" - Victories="+str(self.victories)+" - Losses:"+str(self.losses))
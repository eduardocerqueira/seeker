#date: 2023-05-12T16:50:14Z
#url: https://api.github.com/gists/242123fdda868b695101100504ad25f1
#owner: https://api.github.com/users/odavidsons

#Play rock paper scissors against this program
import random

while 1: #Handle user input for the number of rounds played
    try:
        rounds = int(input("How many rounds do you wanna play? "))
        break
    except ValueError:
        print("That's not a valid number")
        continue
options = ["Rock","Paper","Scissors"] #List of options for the CPU

def checkWinner(cpu_choice,player_choice): #Check who won the round and return their name
    winner = ""
    if cpu_choice == player_choice: winner = "Draw"
    elif cpu_choice =="Rock" and player_choice == "Scissors": winner = "Rock"
    elif cpu_choice =="Scissors" and player_choice == "Rock": winner = "Rock"
    elif cpu_choice =="Paper" and player_choice == "Scissors": winner = "Scissors"
    elif cpu_choice =="Scissors" and player_choice == "Paper": winner = "Scissors"
    elif cpu_choice =="Scissors" and player_choice == "Rock": winner = "Rock"
    elif cpu_choice =="Rock" and player_choice == "Scissors": winner = "Rock"
    elif cpu_choice =="Rock" and player_choice == "Paper": winner = "Paper"
    elif cpu_choice =="Paper" and player_choice == "Rock": winner = "Paper"
    
    if cpu_choice == winner: 
        winner = "CPU"
    elif player_choice == winner: 
        winner = "Player"
    return winner   

#Main program
cpu_points = 0
player_points = 0
for i in range(rounds): 
    cpu_choice = random.choice(options)
    print("------------------------------------------------------")
    player_choice = ""
    while player_choice != "Rock" and player_choice != "Paper" and player_choice != "Scissors": #Force user to write only a valid option
        player_choice = input("What will you play? Rock/Paper/Scissors\n")
        print("------------------------------------------------------")
    print(f"The CPU chose {cpu_choice}...")
    round_winner = checkWinner(cpu_choice, player_choice)
    if round_winner != "Draw": print(f"{round_winner} won the round!") #Print who won
    else: print("This round is a draw!")
    if round_winner == "CPU": cpu_points = cpu_points + 1 #Attribute points accordingly
    elif round_winner == "Player": player_points = player_points + 1
    print(f"Score: CPU {cpu_points}:{player_points} Player") #Current score message
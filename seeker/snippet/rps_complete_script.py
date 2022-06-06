#date: 2022-06-06T17:05:26Z
#url: https://api.github.com/gists/5a614b4c3fc05dfd54922245cfc1963a
#owner: https://api.github.com/users/Noble-47

class RPS:
    options = {"S": "P", "P": "R", "R": "S"}

    def __init__(self, pick, name):
        self.pick = pick
        self.name = name

    def __gt__(self, x):
        # S > P  P > R R > S
        if x.pick == self.options[self.pick]:
            return True
        else:
            return False

    def __repr__(self):
        return f"RPS({self.pick}, {self.name})"

    def __str__(self):
        return f"{self.name}"


option_list = ["R", "P", "S"]
mapping = {
    "R": RPS(pick="R", name="Rock"),
    "P": RPS(pick="P", name="Papper"),
    "S": RPS(pick="S", name="Scissors"),
}


def play():
    print(
        'Enter your choice ("R" for Rock, "P" for Paper, "S" for Scissors) : ', end=""
    )
    user_pick = input().upper()
    # validate user's pick
    if user_pick in option_list:
        comp_pick = random.choice(option_list)
        return evaluate_winner(user_pick, comp_pick)

    else:
        print(
            f"{user_pick} is an Invalid option. \
         \nPlease select either 'R', 'P' or 'S'"
        )


def evaluate_winner(user_pick, comp_pick):
    p1 = mapping.get(user_pick, None)
    p2 = mapping.get(comp_pick, None)
    # check that p1 and p2 has values

    if p1.pick == p2.pick:
        # Its a tie
        print(
            f"player({p1.name}) : CPU({p2.name}) \
                \nIt's a Tie"
        )
        return False

    if p1 > p2:
        # p1 wins
        print(
            f"player({p1.name}) : CPU({p2.name}) \
                \nPlayer Wins!!!"
        )
    else:
        # p2 wins
        print(
            f"player({p1.name}) : CPU({p2.name}) \
                \nCPU Wins!!!"
        )
    return True


def game():
    print("Welcome To My Python Implementation Of Rock Paper Scissors")
    winner = False
    while not winner:
        winner = play()
	print("Existing....")

if __name__ == "__main__":
    game()

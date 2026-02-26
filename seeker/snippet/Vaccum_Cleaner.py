#date: 2026-02-26T17:39:26Z
#url: https://api.github.com/gists/09f676538a6c4fc42dfc7fdf15ba6518
#owner: https://api.github.com/users/nitflame

class VacuumCleaner:
    def __init__(self):
        self.location = 'A'

    def suck(self):
        return f"Sucking dirt in room {self.location}."

    def move_left(self):
        self.location = 'A'
        return "Moving left to room A."

    def move_right(self):
        self.location = 'B'
        return "Moving right to room B."


def main():
    cleaner = VacuumCleaner()

    print("Dynamic Vacuum Cleaner Agent")
    print("Rooms: A, B | Status: clean, dirty")
    print("Type 'exit' anytime to stop.\n")

    while True:
        room = input("Enter current room (A/B): ").strip().upper()
        if room == 'EXIT':
            break

        status = input("Enter room status (clean/dirty): ").strip().lower()
        if status == 'exit':
            break

        # update agent perception
        cleaner.location = room

        # agent decision
        if status == 'dirty':
            action = cleaner.suck()
        else:
            if cleaner.location == 'A':
                action = cleaner.move_right()
            else:
                action = cleaner.move_left()

        print("Action:", action)
        print("-" * 30)

    print("Simulation ended.")


if __name__ == "__main__":
    main()
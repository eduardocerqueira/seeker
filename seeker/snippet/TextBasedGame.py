#date: 2024-12-20T16:48:19Z
#url: https://api.github.com/gists/a7f71f916d15d2b272d3565fe35daf65
#owner: https://api.github.com/users/Cds0012

def show_instructions():
    print("Escape from the Crystal Caverns")
    print("Collect items to win the game, or the cavern collapses")
    print("Move commands: go South, go North, go East, go West")
    print("Add to inventory: get 'item name'")

def show_status(current_room, inventory, room_items):
    print(f"You are in the {current_room}")
    print(f"Inventory: {inventory}")
    if room_items.get(current_room):
        print(f"You see a {room_items[current_room]}")

def main():
    rooms = {
        "Entrance": {'South': 'Crystal Lake','item': None},
        "Crystal Lake":{ 'North': 'Entrance', 'South': 'Whispering walls','item': 'Gemstone Compass'},
        "Whispering Walls": {'North': 'Crystal Lake', 'East': 'Obsidian Maze', 'item': 'Gemstone Compass'},
        "Shifting Sands":{'West':'Entrance Cavern', 'South': 'Obsidian Maze', 'item': 'Map Fragment'},
        "Obsidian Maze": {'North': 'Shifting Sands', 'West': 'Whispering Walls', 'East': 'Giant Geode', 'item': 'Crystal Key'},
        "Giant Geode": {'West': 'Obsidian Maze', 'South': 'Emerald Grotto', 'item': 'Lantern'},
        "Emerald Grotto": {'North': 'Giant Geode', 'South': 'Exits Portal', 'item': 'Ancient Tablet'},
        "Exit Portal": {'item': 'Escapes'}
    }
    current_room = "Entrance"
    inventory = []
    room_items = {room: rooms[room]['item'] for room in rooms}
    show_instructions()
    while True:
        show_status(current_room, inventory, room_items)
        move = input("Enter your move:").strip().lower()
        if move.startswith("go "):
            direction = move.split(" ")[1].capitalize()
            if direction in rooms[current_room]:
                current_room = rooms[current_room] [direction]
            else:
                print("You can't go that way!")
        elif move.startswith("get "):
            item_name = move[4].title()
            if room_items.get(current_room) == item_name:
                inventory.append(item_name)
                room_items[current_room] = None
                print(f"{item_name} added to inventory.")
            else:
                print("That item is not here")
        else:
            print("Invalid command")

        if current_room == "Exit Portal" and len(inventory) < 7:
            print("You died! Cavern collapse")
            break
        elif len(inventory) == 7 and current_room == "Exit Portal":
            print("Congratulations! You collected all items and won the game!")
            break

    if __name__ == "__main__":
        main()





























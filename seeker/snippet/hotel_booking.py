#date: 2025-05-27T17:04:15Z
#url: https://api.github.com/gists/885dd5c79cb1cf3f6dccde12d36a0315
#owner: https://api.github.com/users/banuprakashreddy1111

import random
from itertools import combinations

class Room:
    def __init__(self, floor, number, position):
        self.floor = floor
        self.number = number
        self.position = position  # 1-based index from left
        self.occupied = False
        self.booked = False

    def __repr__(self):
        status = "Occupied" if self.occupied else "Booked" if self.booked else "Available"
        return f"{self.number}({status})"

class Hotel:
    def __init__(self):
        self.rooms = self.generate_rooms()

    def generate_rooms(self):
        rooms = []
        for floor in range(1, 10):  # Floors 1 to 9
            for pos in range(1, 11):
                number = floor * 100 + pos
                rooms.append(Room(floor, number, pos))
        for pos in range(1, 8):  # Floor 10: 1001-1007
            number = 1000 + pos
            rooms.append(Room(10, number, pos))
        return rooms

    def reset(self):
        for r in self.rooms:
            r.occupied = False
            r.booked = False

    def random_occupancy(self, count=20):
        self.reset()
        random_rooms = random.sample(self.rooms, count)
        for r in random_rooms:
            r.occupied = True

    def available_rooms(self):
        return [r for r in self.rooms if not r.occupied]

    def display(self):
        print("\nHotel Room Status:")
        for floor in range(10, 0, -1):
            line = f"Floor {floor:>2}: "
            for r in self.rooms:
                if r.floor == floor:
                    if r.occupied:
                        symbol = "X"
                    elif r.booked:
                        symbol = "B"
                    else:
                        symbol = "O"
                    line += f"{r.number}:{symbol} "
            print(line)
        print("Legend: O=Available, X=Occupied, B=Booked\n")

    def book_rooms(self, n):
        self.clear_booking_marks()
        if n < 1 or n > 5:
            print("You can only book between 1 and 5 rooms.")
            return

        available = self.available_rooms()

        # Try booking on same floor
        for floor in range(1, 11):
            floor_rooms = sorted([r for r in available if r.floor == floor], key=lambda r: r.position)
            for i in range(len(floor_rooms) - n + 1):
                group = floor_rooms[i:i+n]
                if len(group) == n:
                    for r in group:
                        r.booked = True
                    print(f"Booked on floor {floor} (same floor optimization).")
                    return

        # Otherwise find optimal combination
        best_combo = None
        best_time = float('inf')

        for combo in combinations(available, n):
            total_time = self.calculate_total_travel_time(combo)
            if total_time < best_time:
                best_time = total_time
                best_combo = combo

        if best_combo:
            for r in best_combo:
                r.booked = True
            print(f"Booked across floors (optimal travel time: {best_time} minutes).")
        else:
            print("Unable to find enough available rooms to book.")

    def calculate_total_travel_time(self, rooms):
        base = rooms[0]
        time = 0
        for r in rooms[1:]:
            vertical = abs(r.floor - base.floor) * 2
            horizontal = abs(r.position - base.position) * 1
            time += vertical + horizontal
        return time

    def clear_booking_marks(self):
        for r in self.rooms:
            r.booked = False


# Run the program
if __name__ == "__main__":
    hotel = Hotel()

    while True:
        hotel.display()
        print("Options:")
        print("1. Book rooms")
        print("2. Generate random occupancy")
        print("3. Reset all")
        print("4. Exit")

        choice = input("Enter choice: ").strip()

        if choice == "1":
            try:
                count = int(input("How many rooms to book (1–5)? "))
                hotel.book_rooms(count)
            except ValueError:
                print("Invalid input.")
        elif choice == "2":
            hotel.random_occupancy()
        elif choice == "3":
            hotel.reset()
        elif choice == "4":
            break
        else:
            print("Invalid choice.")

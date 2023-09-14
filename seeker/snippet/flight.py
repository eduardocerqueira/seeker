#date: 2023-09-14T17:07:41Z
#url: https://api.github.com/gists/7b446ad28621feb9343029219504985b
#owner: https://api.github.com/users/StanleyXisco

"""Model for Airtravel"""

class Flight:

    def __init__(self, number, aircraft):
        # Class Invariant
        if not number[:2].isalpha():
            raise ValueError(f"No airline code in '{number}'")

        if not number[:2].isupper():
            raise ValueError(f"Invalid airline code in '{number}'")

        if not (number[2:].isdigit() and int(number[2:]) <= 9999):
            raise ValueError(f"Invalid route number '{number}'")

        self._number = number
        self._aircraft = aircraft
        rows, seats = self._aircraft.seating_plan()
        self._seating = [None] + [{letter: None for letter in seats} for _ in rows]

    def aircraft_model(self):
        return self._aircraft.model()

    def number(self):
        return self._number

    def allocate_seat(self,seat, passenger):
        """Allocate a seat for a passanger
        Args:
            seat:A seat designator such as "12c" or "21C"

        Raise:
            ValueError: If the seat is unavailable"""
        row, letter = self._parse_seat(seat)

        if self._seating[row][letter] is not None:
            raise ValueError(f"seat {seat} already occupied")

        self._seating[row][letter] = passenger

    def _parse_seat(self, seat):
        rows, seat_letters = self._aircraft.seating_plan()

        letter = seat[-1]
        if letter not in seat_letters:
            raise ValueError(f"Invalid seat letter {letter}")

        row_text = seat[:-1]
        try:
            row = int(row_text)
        except ValueError:
            raise ValueError(f"Invalid seat row {row_text}")

        if row not in rows:
            raise ValueError(f"Invalid row number {row}")

        return row, letter

    def relocate_passenger(self, from_seat, to_seat):
        """Relocate a passenger to a different seat.
            Args:
                from_seat: The existing seat designator for the passenger to be moved.

                to_seat: The new seat designator.
        """
        from_row, from_letter = self._parse_seat(from_seat)
        if self._seating[from_row][from_letter] is None:
            raise ValueError(f"No passenger to relocate in seat {from_seat}")

        to_row, to_letter = self._parse_seat(to_seat)
        if self._seating[to_row][to_letter] is not None:
            raise ValueError(f"Seat {to_seat} already occupied")

        self._seating[to_row][to_letter] = self._seating[from_row][from_letter]
        self._seating[from_row][from_letter] = None

    def num_seat_available(self):
        return sum(sum(1 for s in row.values() if s is None)
                   for row in self._seating
                   if row is not None)

    def make_boarding_cards(self, card_printer):
        for passenger, seat in sorted(self._passenger_seats()):
            card_printer(passenger, seat, self.number(), self.aircraft_model())

    def _passenger_seats(self):
        """An iterable Series of passenger seating locations."""
        row_numbers, seat_letters = self._aircraft.seating_plan()
        for row in row_numbers:
            for letter in seat_letters:
                passenger = self._seating[row][letter]
                if passenger is not None:
                    yield (passenger, f"{row}{letter}")
    
# class Aircraft:
#     """Aircraft comprehensive details"""
#
#     def __init__(self, registration, model,num_rows, num_seats_per_row):
#         self._registration = registration
#         self._model = model
#         self._num_rows = num_rows
#         self._num_seats_per_row = num_seats_per_row
#
#     def registration(self):
#         return self._registration
#
#     def model(self):
#         return self._model
#
#     def seating_plan(self):
#         return (range(1,self._num_rows + 1),
#                 "ABCDEFGHJK"[:self._num_seats_per_row])


class Aircraft:

    def __init__(self, registration):
        self._registration = registration

    def registration(self):
        return self._registration

    def num_seats(self):
        rows, row_seats = self.seating_plan()
        return len(rows) * len(row_seats)
class AirbusA319(Aircraft):

    def model(self):
        return "Airbus 319"

    def seating_plan(self):
        return range(1,23), "ABCDEF"

class Boeing777(Aircraft):

    def model(self):
        return "Boeing 777"

    def seating_plan(self):
        #For simplicity's' sake, we ignore complex
        #seating Arrangement for First Class
        return range(1,56), "ABCDEGHJK"


def console_card_printer(passenger, seat, flight_number, aircraft):
    output = f"| Name: {passenger}"     \
             f" Flight: {flight_number}"    \
             f" Seat: {seat}"                \
             f" Aircraft: {aircraft}"       \
             " |"
    banner = "+" + "-" * (len(output) - 2) + "+"
    border = "|" + " " * (len(output) - 2) + "|"
    lines = [banner, border, output, border, banner]
    card = "\n".join(lines)
    print(card)
    print()

def make_flights():
    f = Flight("BA756", AirbusA319("G-EUPT"))
    f.allocate_seat("12A", "Nwigboji Stanley")
    f.allocate_seat("15F", "Efrida Didia")
    f.allocate_seat("15E", "Andres Iniesta")
    f.allocate_seat("13A", "Adams Imole")
    f.allocate_seat("1A", "Emma Oshomole")
    f.allocate_seat("3A", "David Shongo")

    g = Flight("AF120", Boeing777("F-Ground0"))
    g.allocate_seat("55K", "Joe Biden")
    g.allocate_seat("33G", "Hillary Cliton")
    g.allocate_seat("4B", "Barrack Obama")
    g.allocate_seat("4A", "Sean Michael")

    return f,g
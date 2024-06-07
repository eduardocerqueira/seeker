#date: 2024-06-07T16:53:39Z
#url: https://api.github.com/gists/91042093fc48f87a8d1ac76127d89f20
#owner: https://api.github.com/users/alexander-kazanski

import tkinter as tk
import csv
from datetime import datetime, timedelta

class PlantTracker:
    def __init__(self, master):
        self.master = master
        self.master.title("Hydroponic Plant Tracker")

        # Load seed data from CSV file
        self.seed_data = self.load_seed_data()

        # Load or create plant data file
        self.plant_data_file = "plant_data.txt"
        self.plant_data = self.load_plant_data()

        # Load or create start date data file
        self.start_date_file = "start_date.txt"
        self.start_dates = self.load_start_dates()

        # Create dictionary to store plant type dropdowns
        self.plants = {}

        # Create labels and dropdown menus for each slot
        for layer in range(3):
            tk.Label(master, text=f"Layer {layer+1}").grid(row=layer * 5, column=0, columnspan=18)
            for row in range(4 if layer == 0 else 3):
                for col in range(9):
                    slot = layer * 27 + row * 9 + col + 1
                    label = tk.Label(master, text=f"Slot {slot}:")
                    label.grid(row=layer * 5 + row + 1, column=col * 2, padx=5, pady=5)
                    # Populate dropdown menu with seed names
                    selected_seed = tk.StringVar(master)
                    selected_seed.set(self.plant_data.get(str(slot), ""))
                    dropdown = tk.OptionMenu(master, selected_seed, *self.seed_data.keys(), command=lambda plant_type, slot=slot: self.save_plant_data(plant_type, slot))
                    dropdown.grid(row=layer * 5 + row + 1, column=col * 2 + 1, padx=5, pady=5)
                    self.plants[slot] = selected_seed

        # Add buttons
        self.growth_button = tk.Button(master, text="Update Growth Stage", command=self.update_growth_stage)
        self.growth_button.grid(row=16, column=0, columnspan=18, pady=10)

        # Create labels for plant start date and days to harvest
        self.start_date_labels = {}
        self.days_to_harvest_labels = {}
        for layer in range(3):
            tk.Label(master, text=f"Start Date - End Date").grid(row=layer * 5 + 17, column=0, columnspan=18)
            for row in range(4 if layer == 0 else 3):
                for col in range(9):
                    slot = layer * 27 + row * 9 + col + 1
                    self.start_date_labels[slot] = tk.Label(master, text="", font=('Helvetica', 10))
                    self.start_date_labels[slot].grid(row=layer * 5 + row + 18, column=col * 2, padx=5, pady=5)
                    self.days_to_harvest_labels[slot] = tk.Label(master, text="", font=('Helvetica', 10))
                    self.days_to_harvest_labels[slot].grid(row=layer * 5 + row + 18, column=col * 2 + 1, padx=5, pady=5)

        # Update growth stage immediately to display dates
        self.update_growth_stage()

    def load_seed_data(self):
        seed_data = {}
        with open("seed_data.csv", "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                seed_data[row['name']] = row
        return seed_data

    def load_plant_data(self):
        plant_data = {}
        try:
            with open(self.plant_data_file, "r") as file:
                for line in file:
                    slot, plant_type = line.strip().split(":")
                    plant_data[slot] = plant_type
        except FileNotFoundError:
            pass
        return plant_data

    def save_plant_data(self, plant_type, slot):
        self.plant_data[str(slot)] = plant_type
        self.start_dates[str(slot)] = datetime.now().strftime("%Y-%m-%d")
        with open(self.plant_data_file, "w") as file:
            for slot, plant_type in self.plant_data.items():
                file.write(f"{slot}:{plant_type}\n")
        with open(self.start_date_file, "w") as file:
            for slot, start_date in self.start_dates.items():
                file.write(f"{slot}:{start_date}\n")
        self.update_growth_stage()  # Update growth stage after selecting a plant

    def load_start_dates(self):
        start_dates = {}
        try:
            with open(self.start_date_file, "r") as file:
                for line in file:
                    slot, start_date = line.strip().split(":")
                    start_dates[slot] = start_date
        except FileNotFoundError:
            pass
        return start_dates

    def update_growth_stage(self):
        # Update start date and days to harvest labels for each slot
        for slot, plant_type in self.plant_data.items():
            if plant_type:
                seed_info = self.seed_data[plant_type]
                start_date = self.start_dates.get(str(slot), "")
                days_to_harvest = seed_info['days to harvest']
                days_to_harvest_min, days_to_harvest_max = map(int, days_to_harvest.split(" - "))
                end_date_min = (datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=days_to_harvest_min)).strftime("%b %d")
                end_date_max = (datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=days_to_harvest_max)).strftime("%b %d")
                start_date_str = datetime.strptime(start_date, "%Y-%m-%d").strftime("%b %d")
                self.start_date_labels[int(slot)].config(text=start_date_str)
                self.days_to_harvest_labels[int(slot)].config(text=f"{end_date_min} - {end_date_max}")

def main():
    root = tk.Tk()
    app = PlantTracker(root)
    root.mainloop()

if __name__ == "__main__":
    main()

#date: 2026-01-29T17:19:57Z
#url: https://api.github.com/gists/e2e03716297cd79dbe1cc72492d832d2
#owner: https://api.github.com/users/Noobsaibot004

import csv

# -----------------------------
# 1. CSV bestand inlezen
# -----------------------------
medewerkers = []

with open("medewerkers.csv", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    for rij in reader:
        medewerkers.append(rij)

# -----------------------------
# 2. Filters
# -----------------------------

# Filter 1: Marketing + Rotterdam
marketing_rotterdam = []
for m in medewerkers:
    if m["afdeling"] == "Marketing" and m["woonplaats"] == "Rotterdam":
        marketing_rotterdam.append(m)

# Filter 2: ouder dan 40 + salaris > 3000
ouder_dan_40_en_hoog_salaris = []
for m in medewerkers:
    if int(m["leeftijd"]) > 40 and int(m["salaris"]) > 3000:
        ouder_dan_40_en_hoog_salaris.append(m)

# Filter 3: woonplaats begint met A
woonplaats_met_a = []
for m in medewerkers:
    if m["woonplaats"].startswith("A"):
        woonplaats_met_a.append(m)

# -----------------------------
# 4. Extra filter (eigen keuze)
# Naam begint met S
# -----------------------------
naam_met_s = []
for m in medewerkers:
    if m["naam"].startswith("S"):
        naam_met_s.append(m)

# -----------------------------
# 5. Sorteren (op salaris)
# -----------------------------
gesorteerd_op_salaris = sorted(
    ouder_dan_40_en_hoog_salaris,
    key=lambda x: int(x["salaris"])
)

# -----------------------------
# 3. Resultaten netjes printen
# -----------------------------
def print_resultaten(titel, lijst):
    print("\n" + titel)
    print(f"{'Naam':<15}{'Afdeling':<15}{'Woonplaats':<15}{'Leeftijd':<10}{'Salaris':<10}")
    print("-" * 65)
    for m in lijst:
        print(f"{m['naam']:<15}{m['afdeling']:<15}{m['woonplaats']:<15}{m['leeftijd']:<10}{m['salaris']:<10}")

print_resultaten("Marketing medewerkers in Rotterdam", marketing_rotterdam)
print_resultaten("Medewerkers ouder dan 40 met salaris > 3000", gesorteerd_op_salaris)
print_resultaten("Medewerkers met woonplaats beginnend met A", woonplaats_met_a)
print_resultaten("Medewerkers met naam beginnend met S", naam_met_s)
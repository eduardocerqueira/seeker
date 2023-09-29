#date: 2023-09-29T16:54:47Z
#url: https://api.github.com/gists/d2c9c523ea93434c1a61a5ec064395fc
#owner: https://api.github.com/users/MatthiasBarth

import csv
import datetime
import math
import os.path
import pickle

import pytz
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from prettytable import PrettyTable

# Client-ID-Konfigurationsdatei
CLIENT_CONFIG_FILE = 'credentials.json'
TOKEN_PICKLE_FILE = "**********"

# Kalender API Einstellungen
SCOPES = ['https://www.googleapis.com/auth/calendar.readonly']

# Überprüfe, ob bereits Anmeldeinformationen vorhanden sind
 "**********"i "**********"f "**********"  "**********"o "**********"s "**********". "**********"p "**********"a "**********"t "**********"h "**********". "**********"e "**********"x "**********"i "**********"s "**********"t "**********"s "**********"( "**********"T "**********"O "**********"K "**********"E "**********"N "**********"_ "**********"P "**********"I "**********"C "**********"K "**********"L "**********"E "**********"_ "**********"F "**********"I "**********"L "**********"E "**********") "**********": "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"w "**********"i "**********"t "**********"h "**********"  "**********"o "**********"p "**********"e "**********"n "**********"( "**********"T "**********"O "**********"K "**********"E "**********"N "**********"_ "**********"P "**********"I "**********"C "**********"K "**********"L "**********"E "**********"_ "**********"F "**********"I "**********"L "**********"E "**********", "**********"  "**********"' "**********"r "**********"b "**********"' "**********") "**********"  "**********"a "**********"s "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********": "**********"
        credentials = "**********"
else:
    # OAuth2.0-Authentifizierung
    flow = "**********"
    credentials = flow.run_local_server(port=0)
    # Speichere die Anmeldeinformationen in einer Datei
 "**********"  "**********"  "**********"  "**********"  "**********"w "**********"i "**********"t "**********"h "**********"  "**********"o "**********"p "**********"e "**********"n "**********"( "**********"T "**********"O "**********"K "**********"E "**********"N "**********"_ "**********"P "**********"I "**********"C "**********"K "**********"L "**********"E "**********"_ "**********"F "**********"I "**********"L "**********"E "**********", "**********"  "**********"' "**********"w "**********"b "**********"' "**********") "**********"  "**********"a "**********"s "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********": "**********"
        pickle.dump(credentials, token)

# API-Client aufbauen
service = build('calendar', 'v3', credentials=credentials)

# Deutsche Zeitzone
german_tz = pytz.timezone('Europe/Berlin')

# Frage nach Monat und Jahr
year = int(input("Bitte gib das Jahr ein (z.B. 2023): "))
month = int(input("Bitte gib den Monat ein (1-12): "))

# Erster und letzter Tag des Monats berechnen
start_of_month = datetime.datetime(year, month, 1).isoformat() + 'Z'  # 'Z' zeigt an, dass das Datum in UTC ist
if month == 12:
    end_of_month = datetime.datetime(year + 1, 1, 1).isoformat() + 'Z'
else:
    end_of_month = datetime.datetime(year, month + 1, 1).isoformat() + 'Z'

# Verfügbare Kalender auflisten
calendar_list = service.calendarList().list().execute()
calendars = calendar_list.get('items', [])
print("\nVerfügbare Kalender:")
for i, cal in enumerate(calendars):
    print(f"{i + 1}. {cal['summary']} (ID: {cal['id']})")

# Kalender auswählen
selected = int(input("\nWähle einen Kalender aus (Nummer): ")) - 1
CALENDAR_ID = calendars[selected]['id']
CALENDAR_NAME = calendars[selected]['summary']

# CSV-Dateinamen erstellen
csv_filename = f"{CALENDAR_NAME.replace(' ', '_').replace(':', '_')}-{month}-{year}.csv"

# Tabelle initialisieren
table = PrettyTable()
table.field_names = ["Kalendarname", "Datum", "Terminbezeichnung", "Start des Termins (HH:MM)",
                     "Ende des Termins (HH:MM)", "Task", "Dauer des Termins in Stunden"]

# CSV-Datei initialisieren
csv_file = open(csv_filename, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(
    ["Kalendarname", "Datum", "Terminbezeichnung", "Start des Termins (HH:MM)", "Ende des Termins (HH:MM)", "Task",
     "Dauer des Termins in Stunden"])

# Termine abrufen und in die Tabelle einfügen
events_result = service.events().list(
    calendarId=CALENDAR_ID, timeMin=start_of_month, timeMax=end_of_month,
    maxResults=100, singleEvents=True, orderBy='startTime').execute()
events = events_result.get('items', [])

for event in events:
    start_time = event['start'].get('dateTime')
    end_time = event['end'].get('dateTime')

    if start_time and end_time:
        start_dt = datetime.datetime.fromisoformat(start_time).astimezone(german_tz)
        end_dt = datetime.datetime.fromisoformat(end_time).astimezone(german_tz)

        # Datum extrahieren
        date_str = start_dt.strftime('%Y-%m-%d')

        duration = (end_dt - start_dt).seconds / 3600
        rounded_duration = math.ceil(duration)

        start_time_str = start_dt.strftime('%H:%M')
        end_time_str = end_dt.strftime('%H:%M')

        # Event-Body überprüfen, ob es [Task]="..." gibt
        event_body = event.get('description', '')  # Du kannst auch 'description' durch das entsprechende Feld ersetzen
        task = "No task description"  # Standardwert, wenn kein Task gefunden wird
        task_start = event_body.find("[Task]=\"")
        if task_start != -1:
            task_start += len("[Task]=\"")
            task_end = event_body.find('"', task_start)
            if task_end != -1:
                task = event_body[task_start:task_end]

        # Entfernen von führenden und nachfolgenden Leerzeichen im Task
        task = task.strip()

        table.add_row([CALENDAR_NAME, date_str, event['summary'], start_time_str, end_time_str, task, rounded_duration])
        csv_writer.writerow([CALENDAR_NAME, date_str, event['summary'], start_time_str, end_time_str, task, rounded_duration])

# Tabelle ausgeben
print(table)

# CSV-Datei schließen
csv_file.close()

print(f"Ergebnisse wurden in '{csv_filename}' gespeichert.")

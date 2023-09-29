#date: 2023-09-29T16:53:27Z
#url: https://api.github.com/gists/8d329c10c061f608ae7cbd278aa0a7ff
#owner: https://api.github.com/users/MatthiasBarth

import datetime
import os.path
import pickle

from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Client-ID-Konfigurationsdatei
CLIENT_CONFIG_FILE = 'credentials.json'
TOKEN_PICKLE_FILE = "**********"

# Kalender API Einstellungen
SCOPES = ['https://www.googleapis.com/auth/calendar.readonly']

# Globale Variable für die Anmeldeinformationen
credentials = None

def authenticate_and_list_calendars():
    global credentials  # Mache die Variable credentials global sichtbar
    # Überprüfe, ob bereits Anmeldeinformationen vorhanden sind
 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"o "**********"s "**********". "**********"p "**********"a "**********"t "**********"h "**********". "**********"e "**********"x "**********"i "**********"s "**********"t "**********"s "**********"( "**********"T "**********"O "**********"K "**********"E "**********"N "**********"_ "**********"P "**********"I "**********"C "**********"K "**********"L "**********"E "**********"_ "**********"F "**********"I "**********"L "**********"E "**********") "**********": "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"w "**********"i "**********"t "**********"h "**********"  "**********"o "**********"p "**********"e "**********"n "**********"( "**********"T "**********"O "**********"K "**********"E "**********"N "**********"_ "**********"P "**********"I "**********"C "**********"K "**********"L "**********"E "**********"_ "**********"F "**********"I "**********"L "**********"E "**********", "**********"  "**********"' "**********"r "**********"b "**********"' "**********") "**********"  "**********"a "**********"s "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********": "**********"
            credentials = "**********"
    else:
        # OAuth2.0-Authentifizierung
        flow = "**********"
        credentials = flow.run_local_server(port=0)
        # Speichere die Anmeldeinformationen in einer Datei
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"w "**********"i "**********"t "**********"h "**********"  "**********"o "**********"p "**********"e "**********"n "**********"( "**********"T "**********"O "**********"K "**********"E "**********"N "**********"_ "**********"P "**********"I "**********"C "**********"K "**********"L "**********"E "**********"_ "**********"F "**********"I "**********"L "**********"E "**********", "**********"  "**********"' "**********"w "**********"b "**********"' "**********") "**********"  "**********"a "**********"s "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********": "**********"
            pickle.dump(credentials, token)

    # API-Client aufbauen
    service = build('calendar', 'v3', credentials=credentials)

    # Kalender auflisten
    calendar_list = service.calendarList().list().execute()
    calendars = calendar_list.get('items', [])

    # Zeige verfügbare Kalender und lass den Benutzer einen auswählen
    print("\nVerfügbare Kalender:")
    for i, cal in enumerate(calendars):
        print(f"{i + 1}. {cal['summary']} (ID: {cal['id']})")

    selected = int(input("\nWähle einen Kalender aus (Nummer): ")) - 1
    calendar_id = calendars[selected]['id']

    return calendar_id

def list_all_events_of_month(calendar_id, year, month):
    # Erster und letzter Tag des angegebenen Monats berechnen
    start_of_month = datetime.datetime(year, month, 1).isoformat() + 'Z'  # 'Z' zeigt an, dass das Datum in UTC ist
    if month == 12:
        end_of_month = datetime.datetime(year + 1, 1, 1).isoformat() + 'Z'
    else:
        end_of_month = datetime.datetime(year, month + 1, 1).isoformat() + 'Z'

    # Aktuelles Datum
    now = datetime.datetime.utcnow().isoformat() + 'Z'

    # API-Client aufbauen
    service = build('calendar', 'v3', credentials=credentials)

    # Ereignisse abrufen
    events_result = service.events().list(
        calendarId=calendar_id, timeMin=start_of_month, timeMax=end_of_month,
        maxResults=10, singleEvents=True, orderBy='startTime').execute()
    events = events_result.get('items', [])

    # Liste für Ereignisse erstellen
    zeit_events = []

    for event in events:
        organizer = event.get('organizer', {}).get('email', '')
        attendees = event.get('attendees', [])

        # Liste der Teilnehmer-Emails erstellen
        attendee_emails = [attendee.get('email', '') for attendee in attendees]

        # Überprüfe, ob die Domain in Organisator oder Teilnehmern vorkommt
        is_domain_match = (
            domain in organizer or
            any(domain in attendee for attendee in attendee_emails)
        )

        # Datum und Name des Ereignisses
        event_date = event['start'].get('dateTime', event['start'].get('date'))
        event_name = event.get('summary', 'Unbenanntes Ereignis')

        if is_domain_match:
            # Mit '*' markieren, wenn die Domain übereinstimmt
            zeit_events.append((f"* Datum: {event_date}", f"* Ereignis: {event_name}", f"* Organisator: {organizer}", f"* Teilnehmer: {', '.join(attendee_emails)}"))
        else:
            zeit_events.append((f"Datum: {event_date}", f"Ereignis: {event_name}", f"Organisator: {organizer}", f"Teilnehmer: {', '.join(attendee_emails)}"))

    return zeit_events

# Verwendungsbeispiel:
selected_calendar_id = authenticate_and_list_calendars()
domain = input("Geben Sie die Domain ein, nach der Sie suchen möchten: ")
year = int(input("Geben Sie das Jahr ein, in dem Sie suchen möchten: "))
month = int(input("Geben Sie den Monat ein, in dem Sie suchen möchten (1-12): "))
zeit_events = list_all_events_of_month(selected_calendar_id, year, month)
if zeit_events:
    print(f"Ereignisse im {month}.{year}:")
    for event in zeit_events:
        print(event)
else:
    print(f"Keine Ereignisse im {month}.{year} gefunden.")

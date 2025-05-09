#date: 2025-05-09T16:53:08Z
#url: https://api.github.com/gists/122d465197c8a9841a03a6cfa95acb1d
#owner: https://api.github.com/users/ProfAndreaPollini

import requests
import json
from datetime import datetime, timezone
import pytz # Per una gestione robusta dei fusi orari

# Coordinate geografiche approssimative per Brescia
BRESCIA_LATITUDE = 45.5389
BRESCIA_LONGITUDE = 10.2207
BRESCIA_TIMEZONE = "Europe/Rome"

def get_solar_irradiance_brescia():
    """
    Recupera i dati di irraggiamento solare attuali e previsti per oggi per Brescia
    utilizzando l'API di Open-Meteo.

    Restituisce:
        dict: Un dizionario contenente i dati orari di irraggiamento (GHI, DNI, DHI)
              e l'ora corrispondente, oppure None se si verifica un errore.
              Include anche le unità di misura.
              Esempio di output per un'ora specifica:
              {
                  "latitude": 45.54,
                  "longitude": 10.22,
                  "timezone": "Europe/Rome",
                  "elevation_m": 149.0,
                  "units": { ... },
                  "current_or_latest_past_hour_data": {
                      "time_iso8601": "2024-05-10T14:00",
                      "ghi_w_m2": 750.5,
                      "dni_w_m2": 850.2,
                      "dhi_w_m2": 150.8
                  },
                  "all_today_hourly_data": [ ...lista di dati orari... ]
              }
    """
    api_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": BRESCIA_LATITUDE,
        "longitude": BRESCIA_LONGITUDE,
        "hourly": "shortwave_radiation,direct_normal_irradiance,diffuse_radiation,direct_radiation",
        "timezone": BRESCIA_TIMEZONE,
        "forecast_days": 1,
    }

    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()
        data = response.json()

        if not data or "hourly" not in data or "hourly_units" not in data:
            print("Errore: Dati mancanti o struttura inattesa nella risposta dell'API.")
            return None

        hourly_data = data["hourly"]
        times = hourly_data.get("time", [])
        ghi_values = hourly_data.get("shortwave_radiation", [])
        dni_values = hourly_data.get("direct_normal_irradiance", [])
        dhi_values = hourly_data.get("diffuse_radiation", [])

        if not all([times, ghi_values, dni_values, dhi_values]):
            print("Errore: Uno o più set di dati orari sono mancanti.")
            return None

        processed_data = []
        for i in range(len(times)):
            processed_data.append({
                "time_iso8601": times[i],
                "ghi_w_m2": ghi_values[i] if ghi_values[i] is not None else 0,
                "dni_w_m2": dni_values[i] if dni_values[i] is not None else 0,
                "dhi_w_m2": dhi_values[i] if dhi_values[i] is not None else 0,
            })

        # Ottieni il fuso orario di Brescia
        brescia_pytz_timezone = pytz.timezone(BRESCIA_TIMEZONE)
        # Ottieni l'ora attuale, consapevole del fuso orario
        now_brescia_tz_aware = datetime.now(brescia_pytz_timezone)

        current_hour_data = None
        latest_valid_index = -1

        for i, t_str in enumerate(times):
            # 1. Parsa la stringa di tempo dall'API in un datetime naive
            naive_api_datetime = datetime.fromisoformat(t_str)
            # 2. Rendi il datetime consapevole del fuso orario (lo stesso che abbiamo richiesto all'API)
            aware_api_datetime = brescia_pytz_timezone.localize(naive_api_datetime)

            # 3. Ora il confronto è tra due datetime consapevoli del fuso orario
            if aware_api_datetime <= now_brescia_tz_aware:
                latest_valid_index = i
            else:
                break
        
        if latest_valid_index != -1:
            current_hour_data = processed_data[latest_valid_index]
        elif processed_data:
            current_hour_data = processed_data[0]

        return {
            "latitude": data.get("latitude"),
            "longitude": data.get("longitude"),
            "timezone": data.get("timezone"),
            "elevation_m": data.get("elevation"),
            "units": data.get("hourly_units"),
            "current_or_latest_past_hour_data": current_hour_data,
            "all_today_hourly_data": processed_data
        }

    except requests.exceptions.RequestException as e:
        print(f"Errore durante la richiesta all'API Open-Meteo: {e}")
        return None
    except json.JSONDecodeError:
        print("Errore: Impossibile decodificare la risposta JSON dall'API Open-Meteo.")
        return None
    except pytz.exceptions.AmbiguousTimeError as e:
        print(f"Errore di fuso orario ambiguo (possibile durante il cambio ora legale/solare): {e}")
        print("Potrebbe essere necessario gestire questo caso specifico se si verifica frequentemente.")
        return None
    except Exception as e:
        print(f"Si è verificato un errore imprevisto: {e.__class__.__name__} - {e}")
        return None

if __name__ == "__main__":
    print(f"Recupero dei dati di irraggiamento solare per Brescia (Lat: {BRESCIA_LATITUDE}, Lon: {BRESCIA_LONGITUDE})...")
    # Nota: la data di esecuzione reale sarà quella corrente quando si esegue lo script.
    # L'esempio nel prompt era per il 9 maggio 2025.
    print(f"Data e ora correnti (fuso orario di Brescia): {datetime.now(pytz.timezone(BRESCIA_TIMEZONE)).isoformat()}")
    irradiance_data = get_solar_irradiance_brescia()

    if irradiance_data:
        print("\n--- Riepilogo ---")
        print(f"Latitudine: {irradiance_data['latitude']:.4f}, Longitudine: {irradiance_data['longitude']:.4f}")
        print(f"Fuso orario: {irradiance_data['timezone']}, Altitudine: {irradiance_data['elevation_m']} m")
        print("Unità di misura:")
        if irradiance_data['units']:
            for key, value in irradiance_data['units'].items():
                print(f"  {key}: {value}")
        else:
            print("  Unità non disponibili.")


        if irradiance_data["current_or_latest_past_hour_data"]:
            print("\n--- Dati per l'ora corrente o l'ultima ora passata disponibile ---")
            current_data = irradiance_data["current_or_latest_past_hour_data"]
            ghi_unit = irradiance_data.get('units', {}).get('shortwave_radiation', 'W/m²')
            dni_unit = irradiance_data.get('units', {}).get('direct_normal_irradiance', 'W/m²')
            dhi_unit = irradiance_data.get('units', {}).get('diffuse_radiation', 'W/m²')

            print(f"  Ora: {current_data['time_iso8601']}")
            print(f"  GHI (Irraggiamento Globale Orizzontale): {current_data['ghi_w_m2']} {ghi_unit}")
            print(f"  DNI (Irraggiamento Diretto Normale): {current_data['dni_w_m2']} {dni_unit}")
            print(f"  DHI (Irraggiamento Diffuso Orizzontale): {current_data['dhi_w_m2']} {dhi_unit}")
        else:
            print("\n--- Nessun dato disponibile per l'ora corrente o passata più recente ---")

        # print("\n--- Tutti i dati orari per oggi ({len(irradiance_data['all_today_hourly_data'])} voci) ---")
        # for i, hourly_entry in enumerate(irradiance_data["all_today_hourly_data"]):
        #     if i < 5 or i > len(irradiance_data["all_today_hourly_data"]) - 6 : # Mostra solo inizio e fine se lungo
        #         print(f"  Ora: {hourly_entry['time_iso8601']}, GHI: {hourly_entry['ghi_w_m2']}, DNI: {hourly_entry['dni_w_m2']}, DHI: {hourly_entry['dhi_w_m2']}")
        #     elif i == 5:
        #         print("  ...")

    else:
        print("Nessun dato recuperato.")
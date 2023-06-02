#date: 2023-06-02T16:54:27Z
#url: https://api.github.com/gists/858ee314e911aff5af3c8d4053f47aa4
#owner: https://api.github.com/users/RomanMachuga

"""
Funkcjonalność, którą dostarcza moduł:
    - wczytanie datasetu (funkcja, która po podaniu ścieżki wczytuje dane z pliku do listy. Dodatkowo funkcja przyjmuje parametr, określający czy pierwszy wiersz pliku zawiera etykiety kolumn czy nie. Jeżeli tak, to etykiety wczytywane są do oddzielnej listy);
    - wypisanie etykiet (funkcja wypisująca etykiety lub komunikat, że etykiet nie było w danym datasecie);
    - wypisanie danych datasetu (funkcja wypisuje kolejne wiersze datasetu. Bez podania parametrów wypisywany jest cały dataset, ale możliwe też podanie 2 parametrów, które określają przedział, który ma zostać wyświetlony);
    - podział datasetu na zbiór treningowy, testowy i walidacyjny (funkcja przyjmuje 3 parametry określające procentowo jaka część głównego zbioru danych trafia do poszczególnych zbiorów);
    - wypis liczby klas decyzyjnych;
    - wypisz dane dla podanej wartości klasy decyzyjnej (wypisuje wiersze z zadaną wartością klasy decyzyjnej);
    - zapisanie danych do pliku csv (jako parametr przyjmowana jest dowolna lista, która może być podzbiorem datasetu, zmienną przechowującą dane treningowe, itp. Dodatkowo podawana jest nazwa pliku, do którego dane zostaną zapisane).
Funkcjonalność zrealizowano dla pliku bank.csv (https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip)
"""

import csv
import os
import sys
from typing import Any, Dict, List


def wczytanie_datasetu(sciezka: str, czy_sa_etykiety: bool) -> List[List[Any]]:
    """Funkcja, która po podaniu ścieżki (nazwa pliku, jeżeli w tym samym folderze) wczyta dane z pliku do listy (można użyć modułu csv). Dodatkowo funkcja przyjmuje parametr określający czy pierwszy wiersz pliku zawiera etykiety kolumn czy nie. Jeżeli tak, to etykiety wczytywane są do oddzielnej listy"""
    
    dataset = []

    try:
        with open(sciezka, 'r', newline='', encoding='utf-8') as stream:
            reader = csv.reader(stream, delimiter=';')
            # konwertowanie liczb w datasecie z "str" na "int"
            for wiersz in reader:
                for w in wiersz:
                    if w.isdigit() or (w[0] == '-' and w[1:].isdigit()):
                        wiersz[wiersz.index(w)] = int(w)
                dataset.append(wiersz)
            
        wypisanie_etykiet(dataset, czy_sa_etykiety)
    except FileNotFoundError:
        print('\nBrak podanego pliku. Program kończy swoją pracę.')
        sys.exit(1)
    
    return dataset


def wypisanie_etykiet(dataset: List[List[Any]], czy_sa_etykiety: bool) -> None:
    """Funkcja wypisująca etykiety lub komunikat, że etykiet nie było w danym datasecie"""

    if czy_sa_etykiety:
        print(f'\nEtykiety z pliku CSV:\n{dataset[0]}')
        # zapisywanie do pliku CSV
        czy_zapisac_wynik, csv_name = czy_zapisac_do_pliku_csv()
        if czy_zapisac_wynik:
            zapisywanie_do_csv([dataset[0]], csv_name)
    else:
        print('\nBrak etykiet w danym datasecie')
    

def wypisanie_danych_datasetu(dataset: List[List[Any]], wybor_12: str, dataset_poczatek: int, dataset_koniec: int) -> List[List[Any]]:
    """Funkcja wypisuje kolejne wiersze datasetu. Bez podania parametrów wypisywany jest cały dataset, ale możliwe też podanie 2 parametrów, które określają przedział, który ma zostać wyświetlony"""

    if wybor_12 == '1':
        [print(wiersz) for wiersz in dataset]
        wynik_wypisywania = [wiersz for wiersz in dataset]
    else:
        [print(wiersz) for wiersz in dataset[dataset_poczatek : dataset_koniec + 1]]
        wynik_wypisywania = [wiersz for wiersz in dataset[dataset_poczatek : dataset_koniec + 1]]
    
    return wynik_wypisywania


def podzial_datasetu(dataset: List[List[Any]], przedzialy_zbiorow: Dict[str, int], czy_sa_etykiety: bool) -> List[List[Any]]:
    """Funkcja przyjmuje 3 parametry określające procentowo jaka część głównego zbioru danych trafia do poszczególnych zbiorów: 1 - treningowy, 2 - testowy, 3 - walidacyjny"""

    dlugosc_datasetu = len(dataset) - int(czy_sa_etykiety)  # zastosowanie tu i dalej wyrazu "int(czy_sa_etykiety)" pozwala uniknąć konstrukcji if-else, czyli można w taki sposób "rozpoczynać" dataset od 0 (jeśli nie ma nagłówków) lub od 1 (jeśli są)
    
    dlugosc_przed_1 = round(dlugosc_datasetu * przedzialy_zbiorow['1'] / 100)
    dlugosc_przed_2 = round(dlugosc_datasetu * przedzialy_zbiorow['2'] / 100)
    dlugosc_przed_3 = dlugosc_datasetu - dlugosc_przed_1 - dlugosc_przed_2
    print(f'\nDługość każdego zbioru: treningowy - {dlugosc_przed_1}, testowy - {dlugosc_przed_2}, walidacyjny - {dlugosc_przed_3} wierszy')
    
    dataset_tren = dataset[int(czy_sa_etykiety) : dlugosc_przed_1 + int(czy_sa_etykiety)]
    dataset_test = dataset[dlugosc_przed_1 + int(czy_sa_etykiety) : dlugosc_przed_1 + int(czy_sa_etykiety) + dlugosc_przed_2]
    dataset_walid = dataset[dlugosc_przed_1 + int(czy_sa_etykiety) + dlugosc_przed_2 :]

    # faktyczne przedziały testowe
    przed_1_fakt = round(len(dataset_tren) / (len(dataset) - int(czy_sa_etykiety)) * 100, 2)
    przed_2_fakt = round(len(dataset_test) / (len(dataset) - int(czy_sa_etykiety)) * 100, 2)
    przed_3_fakt = round(len(dataset_walid) / (len(dataset) - int(czy_sa_etykiety)) * 100, 2)

    # drukowanie tabeli porównującej zadeklarowane i faktyczne przedizały
    print('\nTrzy datasety są utworzone. Zadeklarowane i faktyczne przediały:')
    print('-' * 46)
    print(f'| przedziały |   1   |   2   |   3   |  suma |')
    print(f'|------------|-------|-------|-------|-------|')
    print(f'| wejście, % |{przedzialy_zbiorow["1"]:6} |{przedzialy_zbiorow["2"]:6} |{przedzialy_zbiorow["3"]:6} |{(przedzialy_zbiorow["1"] + przedzialy_zbiorow["2"] + przedzialy_zbiorow["3"]):6} |')
    print(f'| fakt, %    |{przed_1_fakt:6} |{przed_2_fakt:6} |{przed_3_fakt:6} |{(przed_1_fakt + przed_2_fakt + przed_3_fakt):6} |')
    print('-' * 46)
    return dataset_tren, dataset_test, dataset_walid


def wypis_liczb_klas_decyzyjnych(dataset: List[List[Any]], czy_sa_etykiety: bool, nr_kolumny: int) -> List[List[Any]]:
    """Wypisanie krotek, gdzie pierwsza wartość to wartość klasy (np. nazwa irysa, dla binarnych 0 lub 1 itd.), a druga to liczebność (kardynalność) tej klasy"""

    klasy_decyzyjne = []
    lista_klas = [wiersz[nr_kolumny] for wiersz in dataset[int(czy_sa_etykiety):]]  # lista klas decyzylnych
    lista_klas = sorted(set(lista_klas))  # unikatowe posortowane klasy decyzyjne
    print('\nOto liczby klas decyzyjnych:')
    print(f'{dataset[0][nr_kolumny]:15}:{"ilość":>6}')

    for klasa in lista_klas:
        liczba_klas = 0
        for wiersz in dataset[int(czy_sa_etykiety):]:
            if klasa == wiersz[nr_kolumny]:
                liczba_klas += 1
        
        klasy_decyzyjne.append((klasa, liczba_klas))
        print(f'{klasa:15}:{liczba_klas:6}')
    
    return klasy_decyzyjne, lista_klas


def wypis_danych_dla_klasy_decyzyjnej(dataset: List[List[Any]], czy_sa_etykiety: bool, klasa_decyzyjna: str, nr_kolumny: int) -> List[List[Any]]:
    """Wypisuje wiersze z zadaną wartością klasy decyzyjnej"""

    lista_klasy_decyzyjnej = []

    for wiersz in dataset[int(czy_sa_etykiety):]:
        if wiersz[nr_kolumny] == klasa_decyzyjna:
            print(wiersz)
            lista_klasy_decyzyjnej.append(wiersz)
    
    return lista_klasy_decyzyjnej


def czy_zapisac_do_pliku_csv() -> List[Any]:
    print('\nCzy zapisać dane wynnikowe do pliku CSV? [t/n] ', end='')
    czy_zapisac = tak_nie()
    csv_name = ''

    if czy_zapisac:
        while csv_name == '':
            csv_name = input('Podaj nazwę pliku CSV (bez rozszerzenia): > ')
        
        csv_name += '.csv'
    
    return czy_zapisac, csv_name


def zapisywanie_do_csv(zapisywana_lista: List[Any], csv_name: str) -> None:
    """Jako parametr przyjmowana jest dowolna lista, która może być podzbiorem datasetu, zmienną przechowującą dane treningowe, itp. Dodatkowo podawana jest nazwa pliku, do którego dane zostaną zapisane"""

    with open(csv_name, 'w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerows(zapisywana_lista)


def tak_nie() -> bool:
    """
    1) Zachęta do wpisania odpowiedzi.
    2) Sprawdzenie czy w jakości odpowiedzi wpisano literę t lub n.
    3) Konwertuowanie odpowiedzi [t/n] na bool.
    """

    while True:
        t_n = input('> ').lower().strip()
        if t_n.isalpha():
            if t_n == 't':
                wybor_bool = True
            else:
                wybor_bool = False
            break
        else:
            print('Wpisz t lub n ', end='')
    
    return wybor_bool


def main() -> None:
    os.system('cls')
    # wczytywanie datasetu    
    sciezka = input('Podaj scieżkę do datasetu lub tylko nazwę pliku, jeżeli znajduje się w tym samym folderze > ')
    print('Czy pierwszy wiersz pliku zawiera etykiety kolumn [t/n] ', end='')
    wybor_bool = tak_nie()

    if wybor_bool:
        czy_sa_etykiety = True
    else:
        czy_sa_etykiety = False
    
    dataset = wczytanie_datasetu(sciezka, czy_sa_etykiety)
    
    # wypisywanie datasetu
    print('\nCzy wypisać dataset? [t/n] ', end='')
    wybor_bool = tak_nie()

    if wybor_bool:
        while True:
            wybor_12 = input('Wybierz jedną opcję: 1 - wypisać cały dataset; 2 - wypisać przedział datasetu [1/2] > ').strip()

            if wybor_12 == '1':
                dataset_poczatek = None
                dataset_koniec = None
                break
            elif wybor_12 == '2':
                dataset_poczatek = int(input('Podaj numer początku przedziału: > '))

                while True:
                    dataset_koniec = int(input('Podaj numer końca przedziału: > '))
                    if dataset_koniec < dataset_poczatek:
                        print('Koniec przedziału nie może być mniejszy od początku.')
                    elif dataset_koniec > len(dataset) - 1:
                        print(f'Dataset zawiera {len(dataset) - 1} wierszy. Koniec przedziału nie może być większy.')
                    else:
                        break
                break

        wynik_wypisywania = wypisanie_danych_datasetu(dataset, wybor_12, dataset_poczatek, dataset_koniec)
        # zapisywanie do pliku CSV
        czy_zapisac_wynik, csv_name = czy_zapisac_do_pliku_csv()

        if czy_zapisac_wynik:
            zapisywanie_do_csv(wynik_wypisywania, csv_name)

    # podział datasetu na podzbiory
    print('\nCzy podzielić dataset na podzbiory: 1 - treningowy, 2 - testowy, 3 - walidacyjny? [t/n] ', end='')
    wybor_bool = tak_nie()

    if wybor_bool:
        print('Podaj procentowo jaka część głównego zbioru danych trafi do poszczególnych trzech zbiorów')
        przedzialy_zbiorow = dict.fromkeys(['1', '2', '3'], 0)

        while True:
            for pr_zb in przedzialy_zbiorow:
                przedzialy_zbiorow[pr_zb] = int(input(f'- podzbiór {pr_zb} > ').strip())

            if sum(przedzialy_zbiorow.values()) != 100:
                print('Suma wprowadzonych trzech wartości nie jest równa 100. Sprawdź i wprowadź je ponownie:')
            else:
                break
        
        dataset_tren, dataset_test, dataset_walid = podzial_datasetu(dataset, przedzialy_zbiorow, czy_sa_etykiety)
        # zapisywanie do pliku CSV trzech podzbiorów
        print('\nCzy zapisać otrzymane podzbiory? [t/n] ', end='')
        wybor_bool = tak_nie()

        if wybor_bool:
            dataset_dict = {'1 - treningowy': dataset_tren, '2 - testowy': dataset_test, '3 - walidacyjny': dataset_walid}
            print('Czy zapisać podzbiór:')

            for dts_dt in dataset_dict:
                czy_zapisac_wynik = 't'
                czy_zapisac_wynik = input(f'{dts_dt} [T/n] > ').lower().strip()

                if czy_zapisac_wynik == 't':
                    csv_name = ''
                    while csv_name == '':
                        csv_name = input('Podaj nazwę pliku CSV (bez rozszerzenia): > ')
        
                    csv_name += '.csv'
                    zapisywanie_do_csv(dataset_dict[dts_dt], csv_name)

    # wypis liczby klas decyzyjnych
    print('\nCzy wypisać liczby klas decyzyjnych? [t/n] ', end='')
    wybor_bool = tak_nie()

    if wybor_bool:
        print(f'\nPodaj numer kolumny klasy decyzyjnej (0 - to jest pierwsza kolumna, {len(dataset[0]) - 1} - to jest ostatnia kolumna). Dla analizowanego zbioru rekomendowane kolumny dla wyboru klasy decyzyjnej to są kolumny 0 nr 1, 2 lub 3 ', end='')

        while True:
            try:
                nr_kolumny = int(input('> '))
                if 0 <= nr_kolumny <= len(dataset[0]) - 1:
                    break
                else:
                    print(f'Wpisz numer od 0 do {len(dataset[0]) - 1} ', end='')
            except ValueError:
                print('Błąd wprowadzonej wartości ', end='')
        
        klasy_decyzyjne, lista_klas = wypis_liczb_klas_decyzyjnych(dataset, czy_sa_etykiety, nr_kolumny)
        # zapisywanie do pliku CSV
        czy_zapisac_wynik, csv_name = czy_zapisac_do_pliku_csv()

        if czy_zapisac_wynik:
            zapisywanie_do_csv(klasy_decyzyjne, csv_name)

    # wypis danych dla podanej wartości klasy decyzyjnej
    print('\nCzy wypisać dane dla podanej wartości klasy decyzyjnej? [t/n] ', end='')
    wybor_bool = tak_nie()

    if wybor_bool:
        print('Podaj nazwę klasy decyzyjnej ', end='')

        while True:
            klasa_decyzyjna = input('> ').lower().strip()
            if klasa_decyzyjna not in lista_klas:
                print('Nie ma takiej klasy. Wpisz jeszcze raz ', end='')
            else:
                break
        
        lista_klasy_decyzyjnej = wypis_danych_dla_klasy_decyzyjnej(dataset, czy_sa_etykiety, klasa_decyzyjna, nr_kolumny)
        # zapisywanie do pliku CSV
        czy_zapisac_wynik, csv_name = czy_zapisac_do_pliku_csv()
        
        if czy_zapisac_wynik:
            zapisywanie_do_csv(lista_klasy_decyzyjnej, csv_name)


if __name__ == '__main__':
    main()
    
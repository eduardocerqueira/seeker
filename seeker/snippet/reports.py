#date: 2023-02-10T17:09:06Z
#url: https://api.github.com/gists/3d538551d63ffeb129aeb60860a773bb
#owner: https://api.github.com/users/Agnieszka1993

def import_data(filename='booking.txt'):
    """
    Import data from a file to a list. The columns are marked as follows:
    hotel;arrival_date;booked_nights;adults;children;babies;meal;country;reservation_status;reservation_status_date
    Expected returned data format:
        [["Resort Hotel", "01/22/2017", 4, 2, 0, 0, "YES", "PL", "Check-Out", "09/20/2022"],
        ["City Hotel", "01/22/2017", 2, 2, 0, 0, "NO", "FR", "Cancelled", "09/20/2022"],
        ...]

    :param str filename: optional, name of the file to be imported

    :returns: list of lists representing accomodation booking data
    :rtype: list
    """
    with open(filename, 'r') as file:
        bokkings = []
        for booking in file:
            bokkings.append(booking.strip().split(';'))
        return bokkings


def export_data(bookings, filename='booking.txt', mode='a'):
    """
    Export data from a list to file. If called with mode 'w' it should overwrite
    data in file. If called with mode 'a' it should append data at the end.

    :param list booking: booking data
    :param str filename: optional, name of file to export data to
    :param str mode: optional, file open mode with the same meaning as\
    file open modes used in Python. Possible values: only 'w' or 'a'

    :raises ValueError: if mode other than 'w' or 'a' was given. Error message:
        'Wrong write mode'
    """

     # if mode != ['a', 'w']:
     #     return 'Wrong write mode'
     # else:
     #    with open(filename, mode) as file:
     #        booking = ';'.join(bookings)
     #        file.write(booking + '\n')


def get_rows_by_booking_status(rows, status):
    """
    Get booking rows by status

    :param list: booking data
    :param str status: status to filter by e.g. Canceled, Checked-out

    :raises ValueError: if given status is not present in the list. Error message: 'Status is not present in list'
    :returns: all rows of given status
    :rtype: list
    """
    lista = []
    for row in rows:
        if status == row[8]:
            lista.append(row)
    if len(lista) == 0:
        return 'Status is not present in list'
    return lista


def get_rows_by_date(rows, date_in, date_out):
    """
    Get rows filtred by specific date

    :param list rows: rows data, date in, date out
    :returns: list of booking in date range betwee date_in and date_out
    :rtype: list
    """


def children_number_in_date(rows, date, hotel):
    """
    Thos method should return amount of children in selected date and specific hotel

    :param list rows: booking data, date, hotel name
    :returns: number of chidren
    :rtype: int
    """
    for row in rows:
        if row[2] == date and row[1] == hotel:
            return int(row[4])


def display_reservation(rows, date):
    """
    Method return table representation of reservation in format:

    hotel | check in   | check out  | adults | children | babies | status
    Ibis  | 20/09/2022 | 25/09/2022 | 2      | 0        | 0      | checked-in


    Please get check out date based on arrival_date and booked nights
    """
    pass
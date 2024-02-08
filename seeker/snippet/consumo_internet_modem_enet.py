#date: 2024-02-08T17:10:12Z
#url: https://api.github.com/gists/07ee989efe5cb5d93d208075167714d7
#owner: https://api.github.com/users/danielxdad

#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Calculo de consumo de tiempo de cuenta de Internet por Enet.
    
    Para registrar el tiempo de conexion/desconexion:
    KPPP => Configure... => Accounts => [Connection Name] => Edit => Execute => Upon connect
        echo -n 'C: ' >> $HOME/internet_time_record.txt; date -Iseconds >> $HOME/internet_time_record.txt
        
    KPPP => Configure... => Accounts => [Connection Name] => Edit => Execute => Upon disconnect
        echo -n 'D: ' >> $HOME/internet_time_record.txt; date -Iseconds >> $HOME/internet_time_record.txt
"""
from __future__ import unicode_literals
import sys
import os
import datetime
import re

RECORD_FILE_NAME = os.path.join(os.environ['HOME'], 'internet_time_record.txt')
# Tiempo total de la cuanta para el mes en horas por persona
CUENTA_TIEMPO_TOTAL = 50.0


def is_leap(year):
    """Determina cuando un ano es biciestro o no"""
    return year % 4 == 0 and ((year % 100 != 0) or (year % 400 == 0))


def get_month_days(date):
    days_per_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if isinstance(date, datetime.date) or isinstance(date, datetime.datetime):
        if is_leap(date.year) and date.month == 2:
            return 29
        return days_per_month[date.month]
    else:
        raise TypeError('El parametro debe ser una instancia de datetime.datetime o datetime.date')


def parse_datetime(dt):
    return datetime.datetime.strptime(dt, '%Y-%m-%dT%H:%M:%S')


def main():
    connection_ranges = []
    current_month = datetime.datetime.now()
    
    rest_month_days = get_month_days(current_month) - current_month.day
    
    cnt_pattern = re.compile('(?<=C: )[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}')
    discnt_pattern = re.compile('(?<=D: )[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}')
    
    with open(RECORD_FILE_NAME, 'r') as fd:
        cdt = ddt = None
        for line in fd:
            line = line.strip()
            
            result = cnt_pattern.search(line)
            if cdt is None and result:
                cdt = parse_datetime(result.group(0))
                if cdt.month != current_month.month:
                    cdt = None
                        
            result = discnt_pattern.search(line)
            if cdt and result:
                ddt = parse_datetime(result.group(0))
                
                if ddt.month == current_month.month:
                    connection_ranges.append((cdt, ddt))
                
                cdt = ddt = None
    
    total = float(sum([(ddt - cdt).seconds for cdt, ddt in connection_ranges]))
    horas, r = divmod(total, 3600)
    minutos = r / 60
    
    tiempo_disp = (CUENTA_TIEMPO_TOTAL - (total / 3600)) * 3600.0
    
    tiempo_dis_horas_dias = float(tiempo_disp) / rest_month_days / 3600.0
    
    print 'Consumo en %s:\n\t%d horas, %.2f minutos' % (current_month.strftime('%B'), horas, minutos)
    print '\tTiempo disponible: %.2f horas\n\tTiempo disponible / dia: %.4f horas' % (tiempo_disp / 3600, tiempo_dis_horas_dias)
    

if __name__ == '__main__':
    sys.exit(main())

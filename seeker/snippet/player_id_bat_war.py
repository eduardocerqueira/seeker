#date: 2022-04-15T16:58:28Z
#url: https://api.github.com/gists/3b5c9d82de1437912fed2c2548f1926d
#owner: https://api.github.com/users/victorfrutuoso

player_id_bat_war['full_name'] = player_id_bat_war['full_name'].replace({
    'AJ Pollock':'A.J. Pollock', 
    'Chin-lung Hu':'Chin-Lung Hu',
    'Henderson Alvarez III':'Henderson Alvarez',
    'Hung-Chih Kuo':'Hong-Chih Kuo',
    'Hyun Jin Ryu':'Hyun-Jin Ryu',
    'JB Shuck':'J.B. Shuck',
    'Luis Cruz':'Luis Alfonso Cruz',
    'Luis Rodriguez':'Luis Antonio Rodriguez',
    'Mike Ryan':'Michael Ryan',
    'Mike Difelice':'Mike DiFelice',
    'Osvaldo Martinez': 'Ozzie Martinez',
    'Rick van den Hurk': 'Rick Van Den Hurk',
    'Russ Mitchell':'Russell Mitchell',
    'Val Pascucci':'Valentino Pascucci',
    'Vidal Nuno III': 'Vidal Nuno',
    'Wil Ledezma':'Wilfredo Ledezma',
    ' Jr.':'',
    ' Jr':',',
    ' III':'',
    '-':' ' 
}, regex = True)
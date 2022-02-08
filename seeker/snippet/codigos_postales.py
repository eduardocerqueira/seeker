#date: 2022-02-08T16:51:18Z
#url: https://api.github.com/gists/4b409a741819d572539fd7e5b8928e23
#owner: https://api.github.com/users/3d24rd0

import pandas as pd
from tqdm import tqdm

# https://github.com/inigoflores/ds-codigos-postales-ine-es/blob/master/data/codigos_postales_municipios.csv

df = pd.read_csv(r'codigos_postales_municipios.csv', sep=',', low_memory=False)

def write_with_progress(df):
    with open('postal.dart', 'w') as file:
        file.write("""
            class Data {
                final String? codigoPostal, nombre;

                Data({
                    required this.codigoPostal,
                    required this.nombre,
                });
            }
            const List<Data> list = [
         """)
        with tqdm(total=len(df)) as pbar:
            for i in range(len(df)):
                cp = str(df.iloc[i]['codigo_postal']).rjust(5, '0')
                name = str(df.iloc[i]['municipio_nombre'])

                file.write('const Data(codigoPostal: "'+cp+'", nombre: "'+name+'"), \n')

                pbar.update(1)
                tqdm._instances.clear()
            
            file.write("];")

write_with_progress(df)

#date: 2024-02-29T16:53:58Z
#url: https://api.github.com/gists/8ec8984cf26e8e7323e829528447f1e3
#owner: https://api.github.com/users/cibervicho

# $ cat requirements.txt
# tqdm==4.66.2

import time
from tqdm import tqdm

def instalar_dependencia(dependencia):
    time.sleep(1)

def main():
    dependencias = ["paquete1", "paquete2", "paquete3"]

    with tqdm(total=len(dependencias), ncols=100, colour='green') as pbar:
        for dependencia in dependencias:
            pbar.set_description(f"Instalando {dependencia}")
            instalar_dependencia(dependencia)
            pbar.update(1)


    print("¡Todas las dependencias se han instalado correctamente!")

if __name__ == "__main__":
    main()
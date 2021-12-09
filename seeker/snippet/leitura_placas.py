#date: 2021-12-09T16:53:07Z
#url: https://api.github.com/gists/bfe12da9f1f74228755463493a634c24
#owner: https://api.github.com/users/fabriciogeog

#!/usr/bin/python3

# IMPORTAÇÕES
import cv2
import numpy as np
import pytesseract
import pandas as pd
from datetime import date

## CLASSE
class Veiculo:
    marca = ''
    modelo = ''
    ano = 0
    cor = ''
    placa = ''

    # FUNÇÕES
    def __init__(self, marca, modelo, ano, cor, placa):
        self.marca = marca
        self.modelo = modelo
        self.ano = ano
        self.cor = cor
        self.placa = placa

    def consulta_placas(leitura):
        tabela = pd.read_excel('veiculos_cadastrados.xls')
        busca = tabela.loc[tabela['PLACA'] == leitura]
        return busca

    def foto_placa(foto, placa_lida):
        data = str(date.today())
        caminho = f'fotos_placas\\{placa_lida}{data}' + '.jpg'
        cv2.imwrite(caminho, foto)

    def registro_placas(resultado):
        resultado.to_csv(
            r'registro_placas.csv',
            mode='a',
            sep=';',
            header=False,
            index=False,
        )

    def leitura_tesseract(quadros, placa_cinza):
        caixas = pytesseract.image_to_data(placa_cinza)
        for x, caixa in enumerate(caixas.splitlines()):
            if x != 0:
                caixa = caixa.split()
                if len(caixa) == 12:
                    x, y, l, a = (
                        int(caixa[6]),
                        int(caixa[7]),
                        int(caixa[8]),
                        int(caixa[9]),
                    )
                    placa_lida = str(caixa[11])
                    cv2.rectangle(
                        quadros, (x, y), (x + l, y + a), (255, 0, 0), 2
                    )
                    cv2.putText(
                        quadros,
                        placa_lida,
                        (x, y - 20),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1,
                        (255, 0, 0),
                        1,
                    )
                    return placa_lida

    def ajuste_imagem(imagem):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        foto = np.array(gray)
        return gray, foto


if __name__ == '__main__':

    def captura_video(url):
        video = cv2.VideoCapture(url)
        while True:
            ret, frames = video.read()
            cv2.imshow('Imagem captada', frames)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
        video.release()
        cv2.destroyAllWindows

    captura_video(0)

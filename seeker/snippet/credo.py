#date: 2022-09-29T17:20:37Z
#url: https://api.github.com/gists/7c3df9430bcd02d8242bd7f29ed851fe
#owner: https://api.github.com/users/luizomf

import re

import requests
from bs4 import BeautifulSoup

fiis = [
    "RECT11",
    "HCTR11",
    "BCFF11",
    "GGRC11",
    "HSML11",
    "KNCR11",
    "MXRF11",
    "SDIL11",
    "VGHF11",
    "VINO11",
    "VISC11",
    "XPLG11",
    "KFOF11",
    "TGAR11",
    "KISU11",
    "MGFF11",
    "XPSF11",
    "BTLG11",
    "VILG11",
    "KNSC11",
    "URPR11",
    "DEVA11",
    "MALL11",
    "XPML11",
    "ALZR11",
    "HGRU11",
    "KNRI11",
    "BBPO11",
]

print('FII'.ljust(16), end=' ')
print('Liquidez diária'.ljust(16), end=' ')
print('DY'.ljust(16), end=' ')
print('Rend. 12M'.ljust(16), end=' ')
print('Rentab.'.ljust(16), end=' ')
print('Patrimônio'.ljust(16), end=' ')
print('V. Cota'.ljust(16), end=' ')
print('V. Patrimonial'.ljust(16), end=' ')
print('P/VP'.ljust(16), end=' ')
print()

for fii in fiis:
    url = f"https://www.fundsexplorer.com.br/funds/{fii}"
    response = requests.get(url)
    html = BeautifulSoup(response.text, 'html.parser')

    liquidez_diaria_str = html.select_one(
        '#main-indicators-carousel .carousel-cell:nth-child(1) .indicator-value'
    ).text
    rendimento_12_meses_str = html.select_one(
        '#dividends > div > div > div.section-body > div:nth-child(1) > div > div.table-responsive > table > tbody > tr:nth-child(1) > td:nth-child(5)'
    ).text
    dividend_yield_str = html.select_one(
        '#main-indicators-carousel .carousel-cell:nth-child(3) .indicator-value'
    ).text
    rentabilidade_mes_str = html.select_one(
        '#main-indicators-carousel .carousel-cell:nth-child(6) .indicator-value'
    ).text
    patrimonio_liquido_str = html.select_one(
        '#main-indicators-carousel .carousel-cell:nth-child(4) .indicator-value'
    ).text
    valor_da_cota_str = html.select_one(
        '#stock-price > span.price'
    ).text
    valor_patrimonial_str = html.select_one(
        '#main-indicators-carousel .carousel-cell:nth-child(5) .indicator-value'
    ).text
    p_vp_str = html.select_one(
        '#main-indicators-carousel .carousel-cell:nth-child(7) .indicator-value'
    ).text
    ultimo_rendimento_str = html.select_one(
        '#main-indicators-carousel .carousel-cell:nth-child(2) .indicator-value'
    ).text

    def convert_to_number(string: str):
        string = string.strip()
        clean_string = re.sub(r'[^0-9\,]', '', string).replace(',', '.')
        float_number = float(clean_string)

        if string.endswith('bi'):
            float_number *= 1_000_000_000
        elif string.endswith('mi'):
            float_number *= 1_000_000

        return round(float_number, 2)

    liquidez_diaria = convert_to_number(liquidez_diaria_str)
    rendimento_12_meses = round(
        convert_to_number(rendimento_12_meses_str) / 12, 2)
    rentabilidade_mes = convert_to_number(rentabilidade_mes_str)
    patrimonio_liquido = convert_to_number(patrimonio_liquido_str)
    valor_da_cota = convert_to_number(valor_da_cota_str)
    valor_patrimonial = convert_to_number(valor_patrimonial_str)
    p_vp = round(valor_da_cota / valor_patrimonial, 2)
    dividend_yield = convert_to_number(dividend_yield_str)
    dy_baseado_em_12_meses = round(
        (rendimento_12_meses / valor_da_cota) * 100, 2)
    ultimo_rendimento = convert_to_number(ultimo_rendimento_str)

    print(f'{fii.upper()}'.ljust(16), end=' ')
    print(f'{liquidez_diaria}'.ljust(16), end=' ')
    print(f'{dividend_yield}'.ljust(16), end=' ')
    print(f'{rendimento_12_meses}'.ljust(16), end=' ')
    print(f'{rentabilidade_mes}'.ljust(16), end=' ')
    print(f'{patrimonio_liquido}'.ljust(16), end=' ')
    print(f'{valor_da_cota}'.ljust(16), end=' ')
    print(f'{valor_patrimonial}'.ljust(16), end=' ')
    print(f'{p_vp}'.ljust(16), end=' ')
    print()

    # print(f'{liquidez_diaria=}')
    # print(f'{rendimento_12_meses=}')
    # print(f'{ultimo_rendimento=}')
    # print(f'{dividend_yield=}')
    # print(f'{dy_baseado_em_12_meses=}')
    # print(f'{rentabilidade_mes=}')
    # print(f'{patrimonio_liquido=}')
    # print(f'{valor_da_cota=}')
    # print(f'{valor_patrimonial=}')
    # print(f'{p_vp=}')

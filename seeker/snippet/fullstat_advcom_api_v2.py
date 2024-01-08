#date: 2024-01-08T16:59:09Z
#url: https://api.github.com/gists/a057a0b6733f0a8c72e0bdea1be649f0
#owner: https://api.github.com/users/MikyPo

# Импортируем библиотеки
import requests
import json
import time
import pandas as pd
import datetime
from datetime import timedelta
import locale
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Читаем token из txt-файла
 "**********"w "**********"i "**********"t "**********"h "**********"  "**********"o "**********"p "**********"e "**********"n "**********"( "**********"' "**********"s "**********"r "**********"c "**********"/ "**********"a "**********"d "**********"v "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"w "**********"b "**********". "**********"t "**********"x "**********"t "**********"' "**********", "**********"  "**********"' "**********"r "**********"' "**********") "**********"  "**********"a "**********"s "**********"  "**********"f "**********": "**********"
    wb_token = "**********"
    
# Переключаем локаль на русскую
locale.setlocale(locale.LC_ALL, 'ru_RU.utf8')

# Функция получает cписки кампаний
# Описание метода в API Wildberries: https://openapi.wb.ru/promotion/api/ru/#tag/Prodvizhenie/paths/~1adv~1v1~1promotion~1count/get
def wb_all_adv_list():
    request_data = []
    adv_list_request = requests.get(
        f"https://advert-api.wb.ru/adv/v1/promotion/count",
        headers={"Authorization": "**********"
    )
    if adv_list_request.status_code == 200:
        request_data = adv_list_request.json()

        # Получаем список всех advertId в одной колонке
        advert_ids = [advert["advertId"] for sublist in request_data["adverts"] for advert in sublist["advert_list"]]

        # Получаем значения type и status в соответствии с advertId
        type_status_dict = {}
        for sublist in request_data["adverts"]:
            for advert in sublist["advert_list"]:
                type_status_dict[advert["advertId"]] = {"type": sublist["type"], "status": sublist["status"]}

        # Создаем список словарей для каждого advertId
        data = []
        for advert_id in advert_ids:
            data.append({"advertId": advert_id, "type": type_status_dict[advert_id]["type"], "status": type_status_dict[advert_id]["status"]})

        return pd.DataFrame(data)

    else:
        print(f"Ошибка при выполнении запроса. Код статуса: {adv_list_request.status_code}")

# Список всех ID рекламных кампаний
# list_campaign = wb_all_adv() - старая функция не работает
list_campaign = wb_all_adv_list()

# Проверяем чего получили
list_campaign.info()
display(list_campaign)

# Фильтруем кампании только со статусом: 7 - кампания завершена, 9 - идут кампании или 11 - кампания на паузе
camp_to_inf = list_campaign.loc[list_campaign['status'].isin([7, 9, 11])]

# Оставляем только ID кампании
camp_inf_ID = camp_to_inf.drop(['type', 'status'], axis=1)
print(camp_inf_ID)

# Функция получения статистики по рекламным кампаниям
# Описание метода в API Wildberries: https://openapi.wb.ru/promotion/api/ru/#tag/Statistika/paths/~1adv~1v1~1fullstats/post
def adv_wb_info(list_campaign, d_start, d_end):
    request_data = pd.DataFrame()

    for index, row in list_campaign.iterrows():
        id_campaign = row['advertId']
        while True:
            apiUrl = 'https://advert-api.wb.ru/adv/v2/fullstats'
            param = {
                "id": int(id_campaign),
                "interval": {
                    "begin": d_start,
                    "end": d_end
                }
            }
            adverts_data = [param]
            headers = {
                'Authorization': "**********"
                'Content type': 'application/json',
            }

            adv_request = requests.post(
                apiUrl, headers=headers, data=json.dumps(adverts_data))
            if adv_request.status_code == 200:
                data = adv_request.json()

                try:
                    # Создаем DataFrame из полученных данных
                    # Берём только данные по дням
                    # Если нужны и др. данные, то см.полученный JSON и пиши доступ по датафрейму как тебе нужно
                    df_buf = pd.DataFrame(data[0]['days'])
                    df_buf['id_campaign'] = id_campaign

                    # Добавляем к общему DataFrame
                    request_data = pd.concat([request_data, df_buf])
                    print(f"{index} Кампания ID={id_campaign} записана в таблицу")
                    break  # Прерываем цикл while при успешном ответе
                except KeyError:
                    print("Ошибка! Не удалось получить данные из JSON")
                    break  # Прерываем цикл while при ошибке в данных
            else:
                    # По ходу работы функции получаем статусы запросов
                    print(f"Ошибка при выполнении запроса. Код статуса: {adv_request.status_code} Кампания ID={id_campaign}")
                    # Если запрос завершился с ошибкой, то через 60 сек.повторяем запрос
                    time.sleep(60)  # Пауза перед следующей попыткой

        # Вставляем паузу на 60 секунд после каждой итерации
        time.sleep(60)
    return request_data

# Дата начала выгрузки
date_start = '2023-01-01'

# Дата конца выгрузки
date_end = '2023-12-31'

# Получение данных
df = adv_wb_info(camp_inf_ID, date_start, date_end)

# Смотрим что получили
df.info()
display(df.head())

# Немного работы с полученными данными
# Определение номера недели
df['№ Недели'] = pd.to_datetime(df['date'], format="%Y-%m-%d").dt.strftime('%W')
df['№ Недели'] = df['№ Недели'].astype(int)

# Функция для определения начала и окончания недели
def find_week(date):
    start_of_week = date - timedelta(days=date.weekday())
    end_of_week = start_of_week + timedelta(days=6)
    return start_of_week, end_of_week
  
# Определение начала и окончания недели
df['Начало недели'], df['Конец недели'] = zip(*df['date'].map(lambda x: find_week(pd.to_datetime(x, format="%Y-%m-%d"))))
df['Неделя'] = df.apply(lambda row: f"{row['Начало недели'].strftime('%d/%m')}-{row['Конец недели'].strftime('%d/%m')}", axis=1)

# Дописываем в df название проданного через РК товара 
df['name'] = df['apps'].apply(lambda x: x[0]['nm'][0]['name'])

# Если df нужно сохранить в excel, то нужно убрать timezone из формата даты
df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
df['Начало недели'] = df_out['Начало недели'].dt.tz_convert(None)
df['Конец недели'] = df_out['Конец недели'].dt.tz_convert(None)

# Делаем имя для файла excel
start_date = df_out['Дата'].min().strftime('%Y-%m-%d')
end_date = df_out['Дата'].max().strftime('%Y-%m-%d')
current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = 'df_adv_from_' + start_date + '_to_' + end_date + '_formed_' + current_datetime + '.xlsx'

# Сохраняем ы excel
df.to_excel(filename, index=False)


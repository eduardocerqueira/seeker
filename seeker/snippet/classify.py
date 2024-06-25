#date: 2024-06-25T16:56:37Z
#url: https://api.github.com/gists/1c665885a56bc482088c9a959f2f956f
#owner: https://api.github.com/users/Evgeny-Egorov-Projects

import requests
import json
import sys


def post_request(url, headers, files=None, json=None):
    """Отправляет POST-запрос и возвращает ответ."""
    response = requests.post(url, headers=headers, files=files, json=json)
    if response.status_code == 200:
        print(f'Запрос выполнен успешно.')
        return response.json()
    else:
        print(f'Ошибка при выполнении запроса.')
        print(response.text)
        return None


def classify_images(model_id, images_path):
    """Классифицирует изображения."""
    url = 'http://nicct:fl_89@ngn.nicct.ru:8080/predict'
    headers = {'accept': 'application/json'}

    with open(images_path, 'rb') as images_file:
        files = {'modelId': (None, str(model_id)), 'images': images_file}
        return post_request(url, headers=headers, files=files)


def load_xml(xml_path):
    """Загружает XML файл."""
    try:
        with open(xml_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print("Ошибка чтения карты пациента. ", e)
        return None



def summarize_diagnosis(xml_data,class_data):
    url = "http://31.129.97.50:8000/utochnenie_diagnoza"
    data = {"xml_card": xml_data, 'nn_diagnosis': class_data}
    return post_request(url=url, headers={'accept': 'application/json'}, json=data)


def run_command_with_params(image_path : str, xml_path : str, model_id=int(7)):
    print('Отсылаем запрос к сервису ИНС')
    class_result = classify_images(model_id, image_path).get("predicts", None)

    for predict in class_result:
        class_result = predict.get('class', None)

    if class_result is None:
        print("Ошибка классификации изображений.")
        return "Сервис не смог диагностировать пациента"

    print('Ответ сервиса ИНС получен. Поставлен диагноз: ', class_result)
    xml_result=load_xml(xml_path)

    if xml_result is None:
        return "Сервис не смог сформировать рекомендации пациенту."

    print('Отсылаем запрос к cервису ПрологЪ рекомендаций для обследования пациентов с заболеваниями ЦНС.')
    diagnosis=summarize_diagnosis(xml_result, class_result)

    if diagnosis is not None:
        print("Рекомендации для пациента сформированы.")
        return diagnosis
    else:
         print("Ошибка при запросе к cервису ПрологЪ рекомендаций для обследования пациентов с заболеваниями ЦНС.")
         return "Сервис не смог сформировать рекомендации пациенту."


if __name__ == "__main__":

    image_path = str(sys.argv[2])
    xml_path = str(sys.argv[1])
    summary = run_command_with_params(image_path=image_path, xml_path=xml_path)
    print(summary)

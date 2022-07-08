#date: 2022-07-08T17:05:20Z
#url: https://api.github.com/gists/3088a78eaaf813a334f47b46d9c0cd84
#owner: https://api.github.com/users/tristobal

import json

from googletrans import Translator
from functools import reduce


def deep_get(dictionary, keys, default=None):
    return reduce(lambda d, key: d.get(key, default) if isinstance(d, dict) else default, keys.split("."), dictionary)


def print_json(dict_):
    print(json.dumps(dict_, indent=2, sort_keys=True))


def append_new_line(file_name, text_to_append):
    with open(file_name, "a+") as file_object:
        file_object.seek(0)
        data = file_object.read(100)
        if len(data) > 0:
            file_object.write("\n")
        file_object.write(text_to_append)


def translate_rolltable(db_file, db_file_new):
    with open(db_file, 'r') as f:
        for line in f:
            json_ = json.loads(line.strip())
            result = translator.translate(json_['name'], dest='es')
            json_['name'] = result.text
            for r in json_['results']:
                result = translator.translate(r['text'], dest='es')
                r['text'] = result.text
            print_json(json_)
            append_new_line(db_file_new, json.dumps(json_, ensure_ascii=False))


def translate_item(db_file, db_file_new):
    with open(db_file, 'r') as f:
        for line in f:
            json_ = json.loads(line.strip())

            result = translator.translate(json_['name'], dest='es')
            json_['name'] = result.text

            result = translator.translate(json_['data']['description']['value'], dest='es')
            json_['data']['description']['value'] = result.text
            try:
                result = translator.translate(json_['data']['source'], dest='es')
                json_['data']['source'] = result.text
            except KeyError as e:
                print(e)
            print_json(json_)
            append_new_line(db_file_new, json.dumps(json_, ensure_ascii=False))


def translate_item_crea(db_file, db_file_new):
    with open(db_file, 'r') as f:
        for line in f:
            json_ = json.loads(line.strip())
            result = translator.translate(json_['name'], dest='es')
            json_['name'] = result.text

            result = translator.translate(json_['data']['description']['value'], dest='es')
            json_['data']['description']['value'] = result.text

            try:
                result = translator.translate(json_['data']['source'], dest='es')
                json_['data']['source'] = result.text
            except (KeyError, TypeError) as e:
                print(e)
            try:
                for s in json_['skills']:
                    try:
                        result = translator.translate(s['name'], dest='es')
                        s['name'] = result.text
                    except Exception as e:
                        print(f'skill - {e}')
                    try:
                        result = translator.translate(s['data']['description']['value'], dest='es')
                        s['data']['description']['value'] = result.text
                    except Exception as e:
                        print(f'skill - {e}')
                    try:
                        result = translator.translate(s['data']['skillName'], dest='es')
                        s['data']['skillName'] = result.text
                    except Exception as e:
                        print(f'skill - {e}')
            except KeyError as e:
                print(json_['name'])
                print(e)

            print_json(json_)
            append_new_line(db_file_new, json.dumps(json_, ensure_ascii=False))


def translate_actor(db_file, db_file_new):
    with open(db_file, 'r') as f:
        for line in f:
            json_ = json.loads(line.strip())

            result = translator.translate(json_['name'], src='french', dest='es')
            json_['name'] = result.text

            try:
                result = translator.translate(json_['data']['biography']['personalDescription']['value'],
                                              src='french', dest='es')
                json_['data']['biography']['personalDescription']['value'] = result.text
            except Exception as e:
                print(f"name:{json_['name']}, e: {e}")

            for i in json_.get('items', []):
                try:
                    result = translator.translate(i['name'], src='french', dest='es')
                    i['name'] = result.text
                except Exception as e:
                    print(f"name:{json_['name']}, e: {e}")

                try:
                    result = translator.translate(i['data']['skillName'], src='french', dest='es')
                    i['data']['skillName'] = result.text
                except Exception as e:
                    print(e)

                try:
                    result = translator.translate(i['data']['specialization'], src='french', dest='es')
                    i['data']['specialization'] = result.text
                except Exception as e:
                    print(e)

                try:
                    result = translator.translate(i['data']['description']['value'], src='french', dest='es')
                    i['data']['description']['value'] = result.text
                except Exception as e:
                    print(f"name:{json_['name']}, e: {e}")

                try:
                    result = translator.translate(i['data']['description']['chat'], src='french', dest='es')
                    i['data']['description']['chat'] = result.text
                except Exception as e:
                    print(e)

            print_json(json_)
            append_new_line(db_file_new, json.dumps(json_, ensure_ascii=False))


def translate_journalentry(db_file, db_file_new):
    with open(db_file, 'r') as f:
        for line in f:
            json_ = json.loads(line.strip())

            result = translator.translate(json_['name'], src='french', dest='es')
            json_['name'] = result.text

            try:
                result = translator.translate(json_['content'], src='french', dest='es')
                json_['content'] = result.text
            except Exception as e:
                pass

            print_json(json_)
            append_new_line(db_file_new, json.dumps(json_, ensure_ascii=False))

translator = Translator()
# db_rolltable = "/home/cristobal.sanchez/apps/foundrydata/Data/modules/coc7-module-fr-toc/packs/fr-compendiums-rolltable.db"
# db_rolltable_new = "/home/cristobal.sanchez/apps/foundrydata/Data/modules/coc7-module-fr-toc/packs/fr-compendiums-rolltable_es.db"
# translate_rolltable(db_rolltable, db_rolltable_new)

# db_items = "/home/cristobal.sanchez/apps/foundrydata/Data/modules/coc7-module-fr-toc/packs/fr-compendiums-item.db"
# db_items_new = "/home/cristobal.sanchez/apps/foundrydata/Data/modules/coc7-module-fr-toc/packs/fr-compendiums-item_es.db"
# translate_item(db_items, db_items_new)

# db_item_crea = "/home/cristobal.sanchez/apps/foundrydata/Data/modules/coc7-module-fr-toc/packs/fr-compendiums-item-crea.db"
# db_items_crea_new = "/home/cristobal.sanchez/apps/foundrydata/Data/modules/coc7-module-fr-toc/packs/fr-compendiums-item-crea_es.db"
# translate_item_crea(db_item_crea, db_items_crea_new)

# db_actor = "/home/cristobal.sanchez/apps/foundrydata/Data/modules/coc7-module-fr-toc/packs/fr-compendiums-actor.db"
# db_actor_new = "/home/cristobal.sanchez/apps/foundrydata/Data/modules/coc7-module-fr-toc/packs/fr-compendiums-actor_es.db"
# translate_actor(db_actor, db_actor_new)

db_journalentry = "/home/cristobal.sanchez/apps/foundrydata/Data/modules/coc7-module-fr-toc/packs/fr-compendiums-journalentry.db"
db_journalentry_new = "/home/cristobal.sanchez/apps/foundrydata/Data/modules/coc7-module-fr-toc/packs/fr-compendiums-journalentry_es.db"
translate_journalentry(db_journalentry, db_journalentry_new)

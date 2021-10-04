#date: 2021-10-04T17:09:32Z
#url: https://api.github.com/gists/fcd66fb706438e98f468be31093d3bb4
#owner: https://api.github.com/users/Radwatch

import requests
import json
import vertica_python

def search_services(**kwargs):

    if 'authority' in kwargs:
        authorities = f'/authorities/{kwargs["authority"]}'
    else:
        authorities = '/'
    if 'key' in kwargs:
        keys = f'/keys/{kwargs["key"]}'
    else:
        keys = '/'

    # json return
    url = f'https://directory.careaware.com/services-directory/registrations/search{authorities}{keys}'
    headers = {'Accept': 'application/json'}
    try:
        print(f'Searching directory')
        r = requests.get(url, headers=headers)
        r.raise_for_status()
    except:
        print(f'Exception raised..Unable to connect to {url}')
        return None

    # return decoded json
    return json.loads(r.text)

def get_domains():
    user = input("Enter your username: ")
    password = input("Enter your password: ")
    conn_info = {'host': 'CERNOCRSVERTDB-QUERY.CERNERASP.com', 'port': 5433, 'user': user, 'password': password,
                 'database': 'Cerner', 'unicode_error': 'strict', 'autocommit': True, 'use_prepared_statements': False,
                 'connection_timeout': 10}
    # using with for auto connection closing after usage
    query = """select distinct domain
                from OLYPRD.CLIENT_PACKAGE_LEVEL 
                where domain not in 
                (select distinct domain 
                FROM 
                OLYPRD.CLIENT_PACKAGE_LEVEL 
                where 
                pkg_desc = 'CareAware MultiMedia' 
                and pkg_nbr < 97816)"""

    print(f'Running query')
    with vertica_python.connect(**conn_info) as connection:
        cur = connection.cursor()
        cur.execute(query)
        return cur.fetchall()


def main():
    # flatten list, it returns a list of lists
    x = [item for sublist in get_domains() for item in sublist]
    domains = []
    for domain in x:
        domains.append(domain.lower())
    # gather all domains with 1.5
    camm_app = search_services(key='urn:cerner:api:mmf-camm-mpage-app-service-1.0')
    # for each entry in the return
    bad_clients = []
    for key in camm_app:
        if 'camm-mpage' in (key["url"]):
            # returns true if domain is found in the authority section
            if not any(search_term in key["authority"] for search_term in domains):
                # add if domain isn't there
                bad_clients.append(key)
    for entry in bad_clients:
        print(entry)


if __name__ == '__main__':
    main()

#date: 2024-11-28T17:04:33Z
#url: https://api.github.com/gists/00e43d327bb07a8b68f4dce6c51b22b0
#owner: https://api.github.com/users/klamba-coursera

import base64
import csv
import requests

def decode_base64(encoded_str):
    """Decode a Base64-encoded string."""
    decoded_bytes = base64.urlsafe_b64decode(encoded_str + '=' * (4 - len(encoded_str) % 4))
    return decoded_bytes.decode('utf-8')

def get_org_info(org_id):
    """Get organization information using the provided org_id."""
    url = f'https://api.coursera.org/api/thirdPartyOrganizations.v1?ids={org_id}'
    headers = {
        'X-Coursera-Source': 'Internal',
        'X-Coursera-Naptime-Fields': 'all',
        'Cookie': (
            'CAUTH=LflhJOPNQk02R0Xv6_n1rIg2IGBLEebj2UCSsJ2Q5_kJFUMQ4KPH1yBh0PrE1NdfnN6AEvEvhm2gzvckmQ4ZvQ.mf6iQnvZn6_GuH5gkp088g.'
            'ubo7LNiG088cGAGxvM4atFkl7CYmnwYZkdvoUVhAiJ5BQ4qy39sbB4lZM7WyZpORVPE7yT9QTg3D6jQrEnUrVA8VlZYOJrBc383lxQVrvbDPotkU_'
            'oJeaisBm4B8Kknh1TT2gPf1JJPC8gM6gu2GFbOGv6Z-W3H3cFFapgCDgIxodUR3Uoagk-Hb6uvOGVNjvwMpleTVoxm_KqL77ByrUGZxh3cFMFvo_nb'
            '9OWbxI7FuGqXEL6YE1OOGdHiG06qtvYrnmccadJbKXOt7Co5sFp4txVZzFyhpV0sXnAvl1umAlyt3yXj-slQaTWJSoMDlvdpibOBORtJdJ9VOd1ROE'
            'OqxsAXYZAceLmHWcRI8Obn_rSSTyXnCHuihEPpAC0xlBpmuA9nsFmYhzWvEMo39jqFSQ4hZhGx_2b0N_6-tB_QpfhjreH4uz-_cVzqIoZbC; COURS'
            'ERIAN=LudFG4NWYXuH9KBJEp1jUtZcrNSnQaF2JJkS7mcgWwBM6BUo_DVvs5MjFyEcFuxGtdZsEp08cy2iXI-BvkGNYg.a6_62ZgLM7FS5Bug-MLuRQ._'
            'CgUDbr8yeOMRmRUZwxGXqFoIsOKEV7vMl-Ik3adV3pTV27Q_SbzEgvNfBjUs5GKzx1vOXOOhGYD0Z6yRkv3_OxZFw3whPEWvrz2Ah52FB5HhSFy60SZS93wPL'
            'aKPMbnbJNeby0tNNCKFijlmdUm7FbADsEO_8CBvUbaz4O9Ee4; CSRF3-Token= "**********"=FPID2.2.Y33hKnYT8vE21bKvf4'
            '3lFlNXFg%2BmmjaQ0nFuHug%2FQAs%3D.1721024913; FPLC=9%2FT1zPenZ07naJTRAKDVVPW1ViMjZPNJI2ohfpfCluZZgtDMkBbVQWEhP%2FuRZkviL7A'
            '4rjjRsJ4niIRX9p%2FgXCJ8qxh7iI8pv8RZfLEpbsIi0yIOyRN4OgFUh4I7vQ%3D%3D; IR_14726=1732778610999%7C0%7C1732778610999%7C%7C; IR_P'
            'I=20b16e80-451a-11ef-8688-fb396faf1ae5%7C1722376033536; IR_gbd=coursera.org; OptanonAlertBoxClosed=2024-11-28T07:23:31.171'
            'Z; OptanonConsent=isGpcEnabled=0&datestamp=Thu+Nov+28+2024+12%3A53%3A31+GMT%2B0530+(India+Standard+Time)&version=202408.1.0'
            '&browserGpcFlag=0&isIABGlobal=false&hosts=&consentId=531c734d-6e66-4771-8e08-de83fbc7dbb2&interactionCount=1&isAnonUser=1&'
            'landingPath=NotLandingPage&groups=C0001%3A1%2CC0004%3A1%2CC0002%3A1%2CC0003%3A1&AwaitingReconsent=false&geolocation=IN%3BDL; __'
            '204r=https%3A%2F%2Fcoursera.atlassian.net%2F; __204u=6388191636-1732778427350; _ga=GA1.1.892328227.1721024913; _ga_7GZ59JSFWQ=GS1.'
            '1.1732793009.7.1.1732793085.0.0.1912261282; _ga_ZCE2Q9YZ3F=GS1.1.1732793009.7.1.1732793085.45.0.0; _gcl_au=1.1.1105922504.1729016310.'
            '1388387831.1730883167.1730883167; _mkto_trk=id: "**********":_mch-coursera.org-1724142450020-92538; _rdt_uuid=1727124034151'
            '.70671fd8-edf2-4b9b-a6d6-45dea6f9adf5; _uetsid=3414cb30ac0411ef80559d69611c7aba|nkf2wx|2|fr9|0|1791; ab.storage.deviceId.6b51'
            '2fd4-04b5-4fd4-8b44-3f482bc8dcf9=g%3A024b63f7-4704-f1ca-8adc-059f9f0a04a5%7Ce%3Aundefined%7Cc%3A1731476484674%7Cl%3A1732778434231;'
            ' ab.storage.sessionId.6b512fd4-04b5-4fd4-8b44-3f482bc8dcf9=g%3A0c33ac1d-1913-021f-03a3-2e2d67013efb%7Ce%3A1732780412291%7Cc%3A17327'
            '78434230%7Cl%3A1732778612291; ab.storage.userId.6b512fd4-04b5-4fd4-8b44-3f482bc8dcf9=g%3A159588263%7Ce%3Aundefined%7Cc%3A173277843'
            '4230%7Cl%3A1732778434231; fs_uid=#ARGC0#43ea5b0e-5c4d-41d5-adad-347da23a7e42:d752f7bb-cafa-481c-8821-2b1c2d9185cb:1730735503869::1#'
            '/1758314155; mutiny.user.token= "**********"=coursera-1724142450283-5a45566b%3A1; PLAY_SESSION=eyJhb' 
            'GciOiJIUzI1NiJ9.eyJkYXRhIjp7Im9hdXRoMl90b2tlbiI6IkpyMVlUR002b1FrUjNmZEdCNkN0c0p6QlJZUmlrTVpuOVNyYXd3Q0lCdDg9In0sIm5iZiI6MTczMjUwNDI5O'
            'SwiaWF0IjoxNzMyNTA0Mjk5fQ.UkUY_u2Cr90tWy4t1XC9RLbTo1-T13g4KhJyo7K5xNk'
        )
    }
    
    response = requests.get(url, headers=headers)
    data = response.json()
    
    try:
        organization = data['elements'][0]
        login_method = organization.get('loginMethod', 'N/A')
        org_name = organization.get('name', 'N/A')
        slug = organization.get('slug', 'N/A')
    except (KeyError, IndexError):
        login_method = 'N/A'
        org_name = 'N/A'
        slug = 'N/A'
    
    return org_name, org_id, login_method, slug

def process_files(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['File Name', 'Org Name', 'Org ID', 'Login Method', 'Slug'])

        for line in infile:
            file_name = line.strip()
            if file_name.endswith('.xml'):
                base64_id = file_name[:-4]  # Remove .xml
                org_id = decode_base64(base64_id)
                org_name, org_id, login_method, slug = get_org_info(org_id)
                writer.writerow([file_name, org_name, org_id, login_method, slug])

# Replace 'output.txt' with the name of your input file and 'final_output.csv' with the desired output file name
process_files('coursera-thirdparty-saml-metadata-filenames.csv', 'final_output.csv')rdparty-saml-metadata-filenames.csv', 'final_output.csv')
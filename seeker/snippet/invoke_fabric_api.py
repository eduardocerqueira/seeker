#date: 2024-03-28T17:05:05Z
#url: https://api.github.com/gists/09d7befcb157011c340c51cb5d4af42f
#owner: https://api.github.com/users/murggu

import requests, json
from requests.adapters import HTTPAdapter, Retry

def invoke_fabric_api_request(method, uri, payload = None):
    
    API_ENDPOINT = "api.fabric.microsoft.com/v1"

    headers = {
        "Authorization": "**********"
        "Content-Type": "application/json"
    }

    try:
        url = f"https://{API_ENDPOINT}/{uri}"
            
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=5, status_forcelist=[502, 503, 504])
        adapter = HTTPAdapter(max_retries=3)
        session.mount('http://', adapter)
        session.mount('https://', adapter)

        response = session.request(method, url, headers=headers, json=payload, timeout=240)

        if response.text is not None:
            response_text_cleaned = response.text.replace('\\r\\n', '')

            response_details = {
                'status_code': response.status_code,
                'response': response_text_cleaned,
                'headers': dict(response.headers.items()) 
            }
            print(json.dumps(response_details, indent=2))

    except requests.RequestException as ex:
        print(ex)RequestException as ex:
        print(ex)
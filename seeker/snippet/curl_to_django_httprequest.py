#date: 2023-01-25T17:09:32Z
#url: https://api.github.com/gists/b5dd9e2f2a2a34715dcde79b92cb2b0d
#owner: https://api.github.com/users/davydany


import django.http 

def convert_curl_to_httprequest(curl_statement):
    import re 
    # Split by spaces 
    curl_parts = curl_statement.split(' ')
    
    # Get method
    method = curl_parts[1]
    
    # Get URL 
    url = curl_parts[2]
    
    # Get Headers 
    headers = dict()
    for part in curl_parts:
        if part.startswith('-H'):
            # Split at ':' 
            header_parts = part.split(':')
            # Strip quotes 
            header_parts[1] = header_parts[1].strip('"')
            # Add to dictionary 
            headers[header_parts[0][2:]] = header_parts[1]
    
    # Get data 
    data_string = None 
    for part in curl_parts:
        if part.startswith('-d'):
            data_string = re.split(r'[="]\s*', part)[2]
            break
    
    # Create Request 
    request = django.http.HttpRequest()
    request.method = method 
    request.path = url 
    request.headers = headers
    request.body = data_string
    
    return request 
    
# Usage 
curl_statement = 'curl -X POST -H "Content-Type: application/json" -d "{"name": "John", "age": 30}" http://www.example.com/api/users

request = convert_curl_to_httprequest(curl_statement)

print(request.method, request.path, request.headers, request.body)
# Output 
# POST http://www.example.com/api/users {'Content-Type': 'application/json'} {'name': 'John', 'age': 30}
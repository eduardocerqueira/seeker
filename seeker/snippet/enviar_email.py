#date: 2021-12-03T17:07:04Z
#url: https://api.github.com/gists/bd405c1f87472fcf0ea5e30b69f2b418
#owner: https://api.github.com/users/allmeidalima

#chamando as libs
from Google import Create_Service
import base64
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.parser import HeaderParser
from email import encoders
import mimetypes
import pandas as pd
import os
import os.path

#Definindo os parametros 
def enviar():
    CLIENT_SECRET_FILE = 'client_secret.json'
    API_NAME = 'gmail'
    API_VERSION = 'v1'
    SCOPES = ['https://mail.google.com/']

    service = Create_Service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, SCOPES)

    #lendo a planilha e iniciando contadora
    #quantidade= len()
    contador = 0
    print ("oxe")

    #while (contador <= quantidade):
    emailMsg = 'Achei essa vaga fazendo webscraping'
    message = MIMEMultipart()
    message['from'] = 'Julia Ingrid <julia.ingridsantos.7@gmail.com>'
    message['to'] = 'julia.ingridsantos.7@gmail.com'
    message['subject'] = 'teste'
    msg= MIMEText(emailMsg)
    message.attach(msg)

    content_type, encoding = mimetypes.guess_type('teste_01.csv')

    print ("opah")

    if content_type is None or encoding is not None:
        content_type = 'application/octet-stream'
        print ("opa")

    main_type, sub_type = content_type.split('/', 1)
    print ("opah2")
    
    fp = open(f'teste_01.csv', 'rb')
    msg = MIMEBase(main_type, sub_type)
    msg.set_payload(fp.read())
    fp.close()
    filename = os.path.basename(f'teste_01.csv')
    msg.add_header('Content-Disposition', 'attachment', filename=filename)
    message.attach(msg)

    print ("opah3")
    raw_string = {'raw': base64.urlsafe_b64encode(message.as_string().encode("utf-8"))}
    print ("opah4")

    messagem = service.users().messages().send(userId='me', body=raw_string).execute()
    print(messagem)
    print ("opah5")

    contador += 1
enviar()
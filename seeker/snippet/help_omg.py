#date: 2021-12-03T17:15:33Z
#url: https://api.github.com/gists/0867a388eed24213c436409293193d7d
#owner: https://api.github.com/users/tetteu777

from flask import Flask, session, app
from flask_sock import Sock
import firebase_admin
import json
from firebase_admin import credentials
from firebase_admin import firestore
import os
from datetime import timedelta

app = Flask(__name__)
app.config['SECRET_KEY'] = 'suitemeet123456789'
port = int(os.environ.get("PORT", 5000))

sock = Sock(app)
cred = credentials.Certificate("serviceAccount.json")
firebase_admin.initialize_app(cred)
db = firestore.client()
doc_ref = db.collection(u'___excluir_fila_pacientes123')


@app.before_request
def make_session_permanent():
    session.permanent = True
    app.permanent_session_lifetime = timedelta(minutes=10)


@sock.route('/test')
def test(ws):
    antigo = -1
    cpf = -1
    doc_watch = ""
    cpfDoMedico = "123"

    #    {"id_do_paciente":"8888",    "nome":"Cleber Da Silva",    "cpf":"88988988911",    "horario_de_chegada":"2010",    "pediatria":true,    "cliente_cartaosim":false }
    def adicionarPacienteNaFila(map, posicao):
        data = {
            u'date': int(map["horario_de_chegada"]),
            u'dispensar': False,
            u'patient_birth_date': u'1978-05-10T00:00:00',
            u'patient_cpf': map["cpf"],
            u'patient_id': int(map["id_do_paciente"]),
            u'patient_name': map["nome"],
            u'pediatria': map["pediatria"],
            u'position': posicao,
            u'redirect': False,
            u'token_agora': u'ger46456ger465g4er65g456er465ger465gre',
            u'triagem_cpf': None,
            u'waiting': True,
            u'cliente_cartaosim': map["cliente_cartaosim"]
        }

        db.collection(u'___excluir_fila_pacientes123').document(map["cpf"]).set(data)
        print("paciente adicionado!")

    def on_snapshot(doc_snapshot, changes, read_time, ):
        print("cpf dessa thread é:" + str(cpf))
        print("esses sao os cpfs do banco:")

        for doc in changes:
            print(doc.document.id)
            waiting = doc.document._data['waiting']
            print(waiting)
            print(type(waiting))

            dispensar = doc.document._data['dispensar']
            print(dispensar)
            print(type(dispensar))

            if str(dispensar) == str(False) and str(waiting) == str(False):
                # db.collection(u'___excluir_fila_pacientes123').document(doc.document.id).delete()
                print('Estranho')
            if doc.document.id == cpf:
                # waiting = doc.document._data['waiting']
                # print(waiting)
                # print(type(waiting))
                #
                # dispensar = doc.document._data['dispensar']
                # print(dispensar)
                # print(type(dispensar))

                if str(dispensar) != str(True):
                    if str(waiting) == str(True):
                        ws.send(
                            ('Posicao do cpf: ' + str(cpf) + " é: " + str(doc.document._data['position'])).encode(
                                'utf-8'))
                    else:
                        ws.send(
                            ("https://ruthless-mouth.surge.sh/?appid=697152d4a92c484aa2e60f832cecffd8&channel=" + str(
                                doc.document._data['patient_id']) + str(
                                doc.document._data['triagem_cpf']) + "&token=").encode('utf-8'))
                else:
                    ws.send(
                        'O paciente foi dispensado'.encode('utf-8'))

                try:
                    triagem_cpf = str(doc.document._data['triagem_cpf'])
                    cpf_do_medico = str(doc.document._data['cpf_do_medico'])
                    cpfDoMedico = cpf_do_medico

                    print(triagem_cpf)
                    print(cpf_do_medico)
                    if cpf_do_medico is not None and triagem_cpf is not None:
                        print("Saiu da stream da Triagem")
                        colecao_de_medicos = db.collection(u'___excluir_medicos123')
                        print(type(colecao_de_medicos))
                        doc_medico = colecao_de_medicos.document(
                            str(doc.document._data['cpf_do_medico']))
                        print(type(doc_medico))

                        # doc_medico3 = doc_medico.get()
                        #
                        # print(doc_medico3)
                        docs = colecao_de_medicos.stream()

                        # colecao_de_pacientes_do_medico = docs[0].reference.collections()[0]

                        # colecao_de_pacientes_do_medico.on_snapshot(on_snapshot_medico)

                        for doc in docs:
                            # List subcollections in each doc
                            for collection_ref in doc.reference.collections():
                                print(collection_ref.parent.path + collection_ref.id)
                                print(collection_ref.parent.path)
                                collection_ref.on_snapshot(on_snapshot_medico)
                                doc_watch.unsubscribe()
                                # se for a coleção
                                # if

                        # colecao_de_pacientes_do_medico = doc_medico.colletion(u'patient_medic')
                        # print(type(colecao_de_pacientes_do_medico))
                        # try:
                        #     colecao_de_pacientes_do_medico.on_snapshot(on_snapshot_medico)
                        # except:
                        #     print("erro ao iniciar stream do medico")
                except:
                    print("coleção dos médicos não existe")

                    #
                    # ouvir a stream da coleção da fila de pacientes do médico que foi encaminhado
                    #       - enviar posição
                    #       - enviar URL da chamada novamente
                    # ou
                    # enviar a mensagem avisando que foi dispensado

    def on_snapshot_medico(doc_snapshot, changes, read_time, ):

        print("saiu oficialmente da stream da triagem e entrou no snapshot do medico")

        for doc in changes:
            print(doc.document.id)
            if doc.document.id == cpf:
                waiting = doc.document._data['waiting']
                print(waiting)
                print(type(waiting))

                dispensar = doc.document._data['dispensar']
                print(dispensar)
                print(type(dispensar))

                if str(dispensar) != str(True):
                    if str(waiting) == str(True):
                        ws.send(
                            ('Posicao do cpf: ' + str(cpf) + " é: " + str(doc.document._data['position'])).encode(
                                'utf-8'))
                    else:
                        print(doc.document._data)

                        ws.send(
                            ("https://ruthless-mouth.surge.sh/?appid=697152d4a92c484aa2e60f832cecffd8&channel=" + str(
                                doc.document._data['patient_id']) + str(
                                doc.document._data['cpf_do_medico']) + "&token=").encode('utf-8'))
                else:
                    ws.send(
                        'O paciente foi dispensado'.encode('utf-8'))

    while True:
        text = ws.receive()
        jsonLoaded = json.loads(text)
        cpf = jsonLoaded["cpf"]
        ws.send('o cpf recebido foi: ' + cpf)

        colecao = db.collection(u'___excluir_fila_pacientes123').get()
        qtdade_pacientes_fila = len(colecao)
        print("A quantidade de pessoas na fila é: " + str(qtdade_pacientes_fila))

        jaEstaNaFila = False
        posicaoAntiga = -99

        for doc in colecao:
            if str(doc.get('patient_cpf')) == str(cpf):
                jaEstaNaFila = True
                posicaoAntiga = doc.get('position')

        if jaEstaNaFila:
            novaPosicao = posicaoAntiga
        else:
            novaPosicao = qtdade_pacientes_fila + 1

        adicionarPacienteNaFila(jsonLoaded, novaPosicao)

        doc_watch = doc_ref.on_snapshot(on_snapshot)

#date: 2023-01-12T17:12:04Z
#url: https://api.github.com/gists/5b00103561dc6065c82de970491fe205
#owner: https://api.github.com/users/matheusneumanndev

from flask import Flask, session, app
from flask_sock import Sock
import firebase_admin
import json
from firebase_admin import credentials
from firebase_admin import firestore
import os
from datetime import timedelta, time, datetime
from time import sleep

# from flask_debugtoolbar import DebugToolbarExtension

app = Flask(__name__)
#
# # the toolbar is only enabled in debug mode:
# app.debug = True
# app.threaded = True
app.config['SECRET_KEY'] = "**********"
# toolbar = DebugToolbarExtension(app)
# port = int(os.environ.get("PORT", 5000))

sock = Sock(app)
cred = credentials.Certificate("serviceAccount.json")
firebase_admin.initialize_app(cred)
db = firestore.client()
doc_ref = db.collection(u'___excluir_fila_pacientes123')


# cpfM = ""


@app.before_request
def make_session_permanent():
    session.permanent = True
    app.permanent_session_lifetime = timedelta(minutes=5)


@sock.route('/')
def test(ws):
    # global cpfM
    antigo = -1
    cpf = -1
    cpfM = ""
    doc_watch = ""
    cpfDoMedico = "123"
    flag = True

    #    {"id_do_paciente":"8888",    "nome":"Joao Da Silva",    "cpf":"88988988911",    "horario_de_chegada":"2010",
    #    "pediatria":true,    "cliente_cartaosim":false, "data_nascimento":"string" }
    def adicionarPacienteNaFila(map, posicao):

        a = datetime.now()

        minute = ""

        if a.minute < 10:
            minute = "0" + str(a.minute)
        else:
            minute = str(a.minute)

        text = str(a.hour) + minute

        inteiro = int(text)

        data_nascimento_preenchida = "data_nascimento" in map

        # try:
        #     teste = map[]
        # except:
        #     print("data_nascimento nao preenchida")

        if data_nascimento_preenchida is False:
            data_nascimento = u'1978-05-10T00:00:00'
        else:
            data_nascimento = map["data_nascimento"]

        data = {
            u'date': inteiro,
            u'dispensar': False,
            u'patient_birth_date': data_nascimento,
            u'patient_cpf': map["cpf"],
            u'patient_id': int(map["id_do_paciente"]),
            u'patient_name': map["nome"],
            u'pediatria': map["pediatria"],
            u'position': posicao,
            u'redirect': False,
            u'token_agora': "**********"
            u'triagem_cpf': None,
            u'waiting': True,
            u'cliente_cartaosim': map["cliente_cartaosim"]
        }

        try:
            db.collection(u'___excluir_fila_pacientes123').document(map["cpf"]).set(data)
            print("paciente adicionado!")
        except:
            print("erro ao adicionar paciente")

    def on_snapshot():

        doc_paciente_cpf_ref = db.collection(u'___excluir_fila_pacientes123').document(str(cpf))

        doc_paciente_cpf = doc_paciente_cpf_ref.get()

        paciente_esta_na_fila = False

        if doc_paciente_cpf.exists:

            doc_dict = doc_paciente_cpf.to_dict()
            print(f'Document data: {doc_dict["patient_name"]}')
            paciente_esta_na_fila = True
        else:
            print(u'No such document triagem!')
            paciente_esta_na_fila = False
            # triagem_medico()

        while paciente_esta_na_fila:

            text = ws.receive(timeout=2)
            if text == 'dispensar':
                try:
                    db.collection(u'___excluir_fila_pacientes123').document(str(cpf)).delete()
                except:
                    print("erro ao paciente tentar se auto dispensar da triagem")
                try:
                    ws.send("O paciente se dispensou")
                except:
                    print("erro no wssend - se dispensou")

                break

            doc_paciente_cpf_ref = db.collection(u'___excluir_fila_pacientes123').document(str(cpf))

            doc = doc_paciente_cpf_ref.get()

            if doc.exists:
                print(f'Document data: {doc.to_dict()["patient_name"]}')
                paciente_esta_na_fila = True
            else:
                print(u'No such document triagem!')
                paciente_esta_na_fila = False

            print("cpf dessa thread e:" + str(cpf))
            jaMandou = False

            waiting = doc.to_dict()['waiting']
            print("waiting = " + str(waiting))

            dispensar = doc.to_dict()['dispensar']
            print("dispensar = " + str(dispensar))

            if str(dispensar) != str(True):
                if str(waiting) == str(True):
                    try:
                        ws.send(
                            ('Posicao do cpf: ' + str(cpf) + " e: " + str(
                                doc.to_dict()['position'])).encode(
                                'utf-8'))
                    except:
                        print("erro ws.send position")

                else:
                    if not jaMandou:
                        # https://ruthless-mouth.surge.sh/
                        try:
                            ws.send(
                                (
                                        "https://chamada-cartaosim.web.app/?appid=697152d4a92c484aa2e60f832cecffd8&channel=" + str(
                                    doc.to_dict()['patient_id']) + str(
                                    doc.to_dict()['triagem_cpf']) + "&token= "**********"
                        except:
                            print("erro ws.send position")

                        jaMandou = True
            else:
                try:
                    ws.send(
                        'O paciente foi dispensado'.encode('utf-8'))
                    break
                except:
                    print("erro ws.send position")

            if doc.to_dict()['position'] < 50:
                sleep(5)
            else:
                sleep(20)

            try:
                doc = doc_paciente_cpf_ref.get()

                triagem_cpf = str(doc.to_dict()['triagem_cpf'])
                cpf_do_medico = str(doc.to_dict()['cpf_do_medico'])
                cpfDoMedico = cpf_do_medico

                print("triagem_cpf = " + str(triagem_cpf))
                print("cpf_do_medico = " + str(cpf_do_medico))

                print("Saiu da stream da Triagem ?????")

                if cpf_do_medico is not None and triagem_cpf is not None:
                    print("Saiu da stream da Triagem")
                    # colecao_de_medicos = db.collection(u'___excluir_medicos123')
                    #                        print(type(colecao_de_medicos))
                    # doc_medico = colecao_de_medicos.document(
                    #     str(doc.to_dict()['cpf_do_medico']))
                    #                       print(type(doc_medico))
                    cpfM = str(doc.to_dict()['cpf_do_medico'])
                    # try:
                    #     docs = colecao_de_medicos.stream()
                    # except:
                    #     print("erro stream medicos")
                    db.collection(u'___excluir_fila_pacientes123').document(str(cpf)).delete()
                    on_snapshot_medico(doc.to_dict()['cpf_do_medico'], cpf)
                    break

                    # for doc in docs:
                    #     # List subcollections in each doc
                    #     for collection_ref in doc.reference.collections():
                    #         try:
                    #             print(collection_ref.parent.path + collection_ref.id)
                    #             #                                print(collection_ref.parent.path)
                    #             # collection_ref.on_snapshot(on_snapshot_medico)
                    #
                    #         except:
                    #             print("linha 191")

            except:
                print("colecao medico nao existe")
                print("------------====-------------")

    def on_snapshot_medico(cpf_medico, cpf_paciente):

        print("saiu oficialmente da stream da triagem e entrou no snapshot do medico")
        jaMandouLink = False

        # sleep(5)

        try:
            doc_medico_cpf_ref = db.collection(u'___excluir_medicos123').document(str(cpf_medico))
            # test_doc = doc_medico_cpf_ref.get()

            # if test_doc.exists:
            #     print("doc do medico existe")
            # else :
            #     print("doc do medico nao existe")

        except:
            print("erro #1")

        try:
            doc = doc_medico_cpf_ref.collection(u'patient_medic').document(str(cpf_paciente)).get()
        except:
            print("erro #2")

        # doc = doc_medico_cpf_ref.collection(u'patient_medic').document(str(cpf_paciente)).get()
        # 
        # sleep(3)

        paciente_esta_na_fila = False

        if doc.exists:
            doc_dict = doc.to_dict()
            print(f'Document data: {doc.to_dict()["patient_name"]}')
            paciente_esta_na_fila = True
        else:
            print(u'No such document medico!')
            paciente_esta_na_fila = False

        while paciente_esta_na_fila:

            doc_medico_cpf_ref = db.collection(u'___excluir_medicos123').document(str(cpf_medico))

            doc = doc_medico_cpf_ref.collection(u'patient_medic').document(str(cpf_paciente)).get()

            if doc.exists:
                print(f'Document data: {doc.to_dict()["patient_name"]}')
                paciente_esta_na_fila = True
            else:
                print(u'No such document medico!')
                paciente_esta_na_fila = False

            waiting = doc.to_dict()['waiting']
            print(str(cpf_paciente) + " waiting medico = " + str(waiting))
            #               print(type(waiting))

            dispensar = doc.to_dict()['dispensar']
            print(str(cpf_paciente) + " dispensar medico = " + str(dispensar))
            #               print(type(dispensar))

            text = ws.receive(timeout=2)
            if text == 'dispensar':
                try:
                    doc_medico_cpf_ref.collection(u'patient_medic').document(str(cpf_paciente)).delete()
                except:
                    print("erro delete")

                try:
                    ws.send("O paciente se dispensou")
                except:
                    print("erro no wssend - se dispensou")

            if str(dispensar) != str(True):
                if str(waiting) == str(True):
                    try:
                        ws.send(
                            ('Posicao do cpf: ' + str(cpf_paciente) + " e: " + str(doc.to_dict()['position'])).encode(
                                'utf-8'))
                        doc_medico_cpf_ref.collection(u'patient_medic').document(str(cpf_paciente)).update(
                            {"control": 777})
                    except:
                        print("erro ws.send")

                else:
                    if not jaMandouLink:
                        print(doc.to_dict())

                        try:
                            ws.send((
                                            "https://chamada-cartaosim.web.app/?appid=697152d4a92c484aa2e60f832cecffd8&channel=" + str(
                                        doc.to_dict()['patient_id']) + str(
                                        doc.to_dict()['cpf_do_medico']) + "&token= "**********"
                        except Exception as e:
                            print(e)
                            print("erro ws.send")

                        jaMandouLink = True
            else:
                try:
                    ws.send('O paciente foi dispensado'.encode('utf-8'))
                    break
                except:
                    print("erro ws.send")

            sleep(2)

    while True:
        if flag:
            try:
                flag = False
                text = ws.receive()
                jsonLoaded = json.loads(text)

                cpf = jsonLoaded["cpf"]
                ws.send('o cpf recebido foi: ' + cpf)

                colecao = db.collection(u'___excluir_fila_pacientes123').get()
                qtdade_pacientes_fila = len(colecao)


                # remover usuários que já foram atendidos da contagem da fila
                for doc in colecao:
                    waiting = doc.to_dict()['waiting']
                    print("waiting = " + str(waiting))
                    if str(waiting) == str(False):
                        qtdade_pacientes_fila = qtdade_pacientes_fila - 1
                # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

                print("A quantidade de pessoas na fila e: " + str(qtdade_pacientes_fila))

                jaEstaNaFila = False
                posicaoAntiga = -99

                for doc in colecao:
                    if str(doc.get('patient_cpf')) == str(cpf):
                        jaEstaNaFila = True
                        posicaoAntiga = doc.get('position')
                        break

                if jaEstaNaFila:
                    novaPosicao = posicaoAntiga
                else:
                    novaPosicao = qtdade_pacientes_fila + 1

                adicionarPacienteNaFila(jsonLoaded, novaPosicao)

                # doc_watch = doc_ref.on_snapshot(on_snapshot)
                on_snapshot()
            except:
                print("erro no ifFlag = true")
        else:
            try:
                text = ws.receive(timeout=2)
                print('text sink ==> ' + str(text))
            except:
                print("erro no receive linha 296")
                break
            if text == "dispensar":
                print("dispensar")
                print("dispensar cpf paciente = " + str(cpf))  # cpf Paciente
                print("dispensar cpf medico =  " + str(cpfM))  # cpf Medico

                try:
                    db.collection(u'___excluir_fila_pacientes123').document(str(cpf)).delete()
                except:
                    print("alguma colecao nao existe")
                try:
                    docu = db.collection(u'___excluir_medicos123').document(str(cpfM))
                    print(docu)
                except:
                    print("erro pegar colecao do medico")

                try:
                    docu2 = docu.collection(u'patient_medic').document(str(cpf))
                except:
                    print(type(docu2))
                    print(docu2)
                    print("erro pegando doc do paciente do medico")
                try:
                    docu2.delete()
                except:
                    print("erro delete")

                try:
                    ws.send("O paciente se dispensou")
                except:
                    print("erro no wssend - se dispensou")
            else:
                try:
                    if novaPosicao is None:
                        ws.send("conexao ativa")
                    else:
                        ws.send(novaPosicao)

                except:
                    print("erro ws.send novaPosicao")
                    break


if __name__ == "__main__":
    app.run(host='0.0.0.0')_name__ == "__main__":
    app.run(host='0.0.0.0')
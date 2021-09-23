#date: 2021-09-23T17:09:05Z
#url: https://api.github.com/gists/79144bf22602a496e1a84e6b36ae6e77
#owner: https://api.github.com/users/josedavila2745

__author__ = 'Julio Junkes'

'''

Calculadora versão 0.2 - Desculpem se o algoritmo não for muito bom,
mas ao invés de consultar algoritmos de calculadora
procurei desenvolver o meu próprio a medida que os problemas iam surgindo.

Ainda há erros, mas está quase pronta.

Implementem como quiserem... Sugestões e correções são bem vindas!

Abraço!

Julio Junkes

'''
from tkinter import *
import ast
import time

w = Tk()
w.title('Python - Calculadora 0.2 - Julio Junkes')
w_width = 400
w_height = 600
w.geometry(str(w_width) + 'x' + str(w_height) + '+'
           + str(int((w.winfo_screenwidth() - w_width) / 2)) + '+'
           + str(int((w.winfo_screenheight() - w_height) / 2)))
w.maxsize(w_width, w_height)
w.minsize(w_width, w_height)

topo_tela = Label(w, height=1)
topo_tela.pack()
topo_tela['text'] = ''
topo_tela['font'] = 'bold 20 bold'

tela = Entry(w, font='bold 50 bold', bg='white', width=8,
             justify='right', relief='sunken')
tela.pack()
tela.insert(0, 0)

sub_tela = Label(w, height=1)
sub_tela.pack()

fr = Frame(w)
fr.pack(side=TOP)

frA = Frame(w)
frA.pack(side=LEFT)

frB = Frame(w)
frB.pack(side=RIGHT)

fr1A = Frame(frA)
fr1A.pack()

fr2A = Frame(frA)
fr2A.pack()

fr3A = Frame(frA)
fr3A.pack()

fr4A = Frame(frA)
fr4A.pack()

fr5A = Frame(frA)
fr5A.pack()

fr1B = Frame(frB)
fr1B.pack()

fr2B = Frame(frB)
fr2B.pack()

fr3B = Frame(frB)
fr3B.pack()

fr4B = Frame(frB)
fr4B.pack()

fr5B = Frame(frB)
fr5B.pack()

acumulado = 0
valor_tela = 0
operacao_anterior = ''
em_operacao = 0
point = 0
horaatual = time.strftime('%H:%M:%S')


def hora():
    global horaatual
    horaatual = time.strftime('%H:%M:%S')
    sub_tela['text'] = horaatual
    sub_tela['font'] = 'bold 20 bold'
    w.after(100, hora)

hora()


def botao_0():
    global em_operacao, operacao_anterior
    text = tela.get()
    tela.delete(0, len(text))

    if text == '0':
        pass
    else:
        if em_operacao != 1 and operacao_anterior != 'igual':
            text += '0'
        else:
            text = '0'
            em_operacao = 0

        tela.insert(0, text)
    if operacao_anterior == 'igual':
        operacao_anterior = ''
    topo_tela['text'] = ''
    if len(tela.get()) > 8:
        text = tela.get()
        tela.delete(len(tela.get())-1)
        topo_tela['text'] = 'máximo: 8 digitos!'




def botao_1():
    global em_operacao, operacao_anterior
    text = tela.get()
    tela.delete(0, len(text))

    if text == '0':
        text = '1'
    else:
        if em_operacao != 1 and operacao_anterior != 'igual':
            text += '1'
        else:
            text = '1'
            em_operacao = 0

    tela.insert(0, text)
    if operacao_anterior == 'igual':
        operacao_anterior = ''
    topo_tela['text'] = ''
    if len(tela.get()) > 8:
        text = tela.get()
        tela.delete(len(tela.get())-1)
        topo_tela['text'] = 'máximo: 8 digitos!'



def botao_2():
    global em_operacao, operacao_anterior
    text = tela.get()
    tela.delete(0, len(text))

    if text == '0':
        text = '2'
    else:
        if em_operacao != 1 and operacao_anterior != 'igual':
            text += '2'
        else:
            text = '2'
            em_operacao = 0

    tela.insert(0, text)
    if operacao_anterior == 'igual':
        operacao_anterior = ''
    topo_tela['text'] = ''
    if len(tela.get()) > 8:
        text = tela.get()
        tela.delete(len(tela.get())-1)
        topo_tela['text'] = 'máximo: 8 digitos!'


def botao_3():
    global em_operacao, operacao_anterior
    text = tela.get()
    tela.delete(0, len(text))

    if text == '0':
        text = '3'
    else:
        if em_operacao != 1 and operacao_anterior != 'igual':
            text += '3'
        else:
            text = '3'
            em_operacao = 0

    tela.insert(0, text)
    if operacao_anterior == 'igual':
        operacao_anterior = ''
    topo_tela['text'] = ''
    if len(tela.get()) > 8:
        text = tela.get()
        tela.delete(len(tela.get())-1)
        topo_tela['text'] = 'máximo: 8 digitos!'


def botao_4():
    global em_operacao, operacao_anterior
    text = tela.get()
    tela.delete(0, len(text))

    if text == '0':
        text = '4'
    else:
        if em_operacao != 1 and operacao_anterior != 'igual':
            text += '4'
        else:
            text = '4'
            em_operacao = 0

    tela.insert(0, text)
    if operacao_anterior == 'igual':
        operacao_anterior = ''
    topo_tela['text'] = ''
    if len(tela.get()) > 8:
        text = tela.get()
        tela.delete(len(tela.get())-1)
        topo_tela['text'] = 'máximo: 8 digitos!'


def botao_5():
    global em_operacao, operacao_anterior
    text = tela.get()
    tela.delete(0, len(text))

    if text == '0':
        text = '5'
    else:
        if em_operacao != 1 and operacao_anterior != 'igual':
            text += '5'
        else:
            text = '5'
            em_operacao = 0

    tela.insert(0, text)
    if operacao_anterior == 'igual':
        operacao_anterior = ''
    topo_tela['text'] = ''
    if len(tela.get()) > 8:
        text = tela.get()
        tela.delete(len(tela.get())-1)
        topo_tela['text'] = 'máximo: 8 digitos!'


def botao_6():
    global em_operacao, operacao_anterior
    text = tela.get()
    tela.delete(0, len(text))

    if text == '0':
        text = '6'
    else:
        if em_operacao != 1 and operacao_anterior != 'igual':
            text += '6'
        else:
            text = '6'
            em_operacao = 0

    tela.insert(0, text)
    if operacao_anterior == 'igual':
        operacao_anterior = ''
    topo_tela['text'] = ''
    if len(tela.get()) > 8:
        text = tela.get()
        tela.delete(len(tela.get())-1)
        topo_tela['text'] = 'máximo: 8 digitos!'


def botao_7():
    global em_operacao, operacao_anterior
    text = tela.get()
    tela.delete(0, len(text))

    if text == '0':
        text = '7'
    else:
        if em_operacao != 1 and operacao_anterior != 'igual':
            text += '7'
        else:
            text = '7'
            em_operacao = 0

    tela.insert(0, text)
    if operacao_anterior == 'igual':
        operacao_anterior = ''
    topo_tela['text'] = ''
    if len(tela.get()) > 8:
        text = tela.get()
        tela.delete(len(tela.get())-1)
        topo_tela['text'] = 'máximo: 8 digitos!'


def botao_8():
    global em_operacao, operacao_anterior
    text = tela.get()
    tela.delete(0, len(text))

    if text == '0':
        text = '8'
    else:
        if em_operacao != 1 and operacao_anterior != 'igual':
            text += '8'
        else:
            text = '8'
            em_operacao = 0

    tela.insert(0, text)
    if operacao_anterior == 'igual':
        operacao_anterior = ''
    topo_tela['text'] = ''
    if len(tela.get()) > 8:
        text = tela.get()
        tela.delete(len(tela.get())-1)
        topo_tela['text'] = 'máximo: 8 digitos!'


def botao_9():
    global em_operacao, operacao_anterior
    text = tela.get()
    tela.delete(0, len(text))

    if text == '0':
        text = '9'
    else:
        if em_operacao != 1 and operacao_anterior != 'igual':
            text += '9'
        else:
            text = '9'
            em_operacao = 0

    tela.insert(0, text)
    if operacao_anterior == 'igual':
        operacao_anterior = ''
    topo_tela['text'] = ''
    if len(tela.get()) > 8:
        text = tela.get()
        tela.delete(len(tela.get())-1)
        topo_tela['text'] = 'máximo: 8 digitos!'

def botao_c():
    global operacao_anterior, acumulado, valor_tela, em_operacao

    operacao_anterior = ''
    acumulado = valor_tela = 0
    tela.delete(0, len(tela.get()))
    tela.insert(0, '0')
    em_operacao = 0


def botao_mult():
    global operacao_anterior, acumulado, valor_tela, em_operacao

    if em_operacao == 0:
        valor_tela = float(tela.get())
        if operacao_anterior == 'soma':
            acumulado += float(valor_tela)
        if operacao_anterior == 'subtracao':
            acumulado -= float(valor_tela)
        if operacao_anterior == 'multiplicacao':
            acumulado *= float(valor_tela)
        if operacao_anterior == 'divisao':
            acumulado /= float(valor_tela)
        if operacao_anterior == '' or operacao_anterior == 'igual':
            acumulado = valor_tela
        if acumulado != 0:
            texto = str(acumulado)
            if (texto[len(texto) - 1]) == '0'\
                    and (texto[len(texto) - 2]) == '.':
                acumulado = ast.literal_eval(texto)
                acumulado = int(acumulado)
            tela.delete(0, len(tela.get()))
            tela.insert(0, str(acumulado)[:8])
        em_operacao = 1
    else:
        pass
    operacao_anterior = 'multiplicacao'


def botao_div():
    global operacao_anterior, acumulado, valor_tela, em_operacao

    if em_operacao == 0:
        valor_tela = float(tela.get())
        if operacao_anterior == 'soma':
            acumulado += float(valor_tela)
        if operacao_anterior == 'subtracao':
            acumulado -= float(valor_tela)
        if operacao_anterior == 'multiplicacao':
            acumulado *= float(valor_tela)
        if operacao_anterior == 'divisao':
            acumulado /= float(valor_tela)
        if operacao_anterior == '' or operacao_anterior == 'igual':
            acumulado = valor_tela
        if acumulado != 0:
            texto = str(acumulado)
            if (texto[len(texto) - 1]) == '0'\
                    and (texto[len(texto) - 2]) == '.':
                acumulado = ast.literal_eval(texto)
                acumulado = int(acumulado)
            tela.delete(0, len(tela.get()))
            tela.insert(0, str(acumulado)[:8])
        em_operacao = 1
    else:
        pass
    operacao_anterior = 'divisao'


def botao_som():
    global operacao_anterior, acumulado, valor_tela, em_operacao

    if em_operacao == 0:
        valor_tela = float(tela.get())
        if operacao_anterior == 'soma':
            acumulado += float(valor_tela)
        if operacao_anterior == 'subtracao':
            acumulado -= float(valor_tela)
        if operacao_anterior == 'multiplicacao':
            acumulado *= float(valor_tela)
        if operacao_anterior == 'divisao':
            acumulado /= float(valor_tela)
        if operacao_anterior == '' or operacao_anterior == 'igual':
            acumulado = valor_tela
        if acumulado != 0:
            texto = str(acumulado)
            if (texto[len(texto) - 1]) == '0'\
                    and (texto[len(texto) - 2]) == '.':
                acumulado = ast.literal_eval(texto)
                acumulado = int(acumulado)
            tela.delete(0, len(tela.get()))
            tela.insert(0, str(acumulado)[:8])
        em_operacao = 1
    else:
        pass
    operacao_anterior = 'soma'


def botao_sub():
    global operacao_anterior, acumulado, valor_tela, em_operacao

    if em_operacao == 0:
        valor_tela = float(tela.get())
        if operacao_anterior == 'soma':
            acumulado += float(valor_tela)
        if operacao_anterior == 'subtracao':
            acumulado -= float(valor_tela)
        if operacao_anterior == 'multiplicacao':
            acumulado *= float(valor_tela)
        if operacao_anterior == 'divisao':
            acumulado /= float(valor_tela)
        if operacao_anterior == '' or operacao_anterior == 'igual':
            acumulado = valor_tela
        if acumulado != 0:
            texto = str(acumulado)
            if (texto[len(texto) - 1]) == '0'\
                    and (texto[len(texto) - 2]) == '.':
                acumulado = ast.literal_eval(texto)
                acumulado = int(acumulado)
            tela.delete(0, len(tela.get()))
            tela.insert(0, str(acumulado)[:8])
        em_operacao = 1
    else:
        pass
    operacao_anterior = 'subtracao'


def botao_point():
    global em_operacao, operacao_anterior, point

    text = tela.get()
    tela.delete(0, len(text))

    for i in text:
        if i == '.':
            point = 1

    if text == '0':
        text = '0.'
    else:
        if em_operacao == 1 or operacao_anterior == 'igual':
            text = '0.'
            em_operacao = 0

        else:
            if point == 0:
                text += '.'

    tela.insert(0, text)
    point = 0
    if operacao_anterior == 'igual':
        operacao_anterior = ''


def botao_equ():
    global operacao_anterior, acumulado, valor_tela

    valor_tela = tela.get()
    if operacao_anterior == 'soma':
        acumulado += float(valor_tela)
    if operacao_anterior == 'subtracao':
        acumulado -= float(valor_tela)
    if operacao_anterior == 'multiplicacao':
        acumulado *= float(valor_tela)
    if operacao_anterior == 'divisao':
        acumulado /= float(valor_tela)

    if operacao_anterior != 'igual':

        texto = str(acumulado)
        if (texto[len(texto) - 1]) == '0'\
                and (texto[len(texto) - 2]) == '.'\
                or str(tela['text']) == '0':
            acumulado = ast.literal_eval(texto)
            acumulado = int(acumulado)
            tela.delete(0, len(valor_tela))
            tela.insert(0, str(acumulado)[:8])
        else:
            texto = str(acumulado)
            tela.delete(0, len(valor_tela))
            tela.insert(0, texto[:8])

    operacao_anterior = 'igual'
    acumulado = 0


def botao_mm():
    alerta = 0
    for i in str(tela.get()):
        if i == '.':
            alerta = 1
    if alerta == 1:
        valor = float(tela.get())
    else:
        valor = int(tela.get())
    valor *= (-1)
    tela.delete(0, len(tela.get()))
    tela.insert(0, valor)


def botao_apaga():
    if tela.get() != '0':
        tela.delete(len(tela.get())-1)
        if tela.get() == '':
            tela.insert(0, 0)


def key(event):
    if event.keysym == '0':
        botao_0()
    if event.keysym == '1':
        botao_1()
    if event.keysym == '2':
        botao_2()
    if event.keysym == '3':
        botao_3()
    if event.keysym == '4':
        botao_4()
    if event.keysym == '5':
        botao_5()
    if event.keysym == '6':
        botao_6()
    if event.keysym == '7':
        botao_7()
    if event.keysym == '8':
        botao_8()
    if event.keysym == '9':
        botao_9()
    if event.keysym == 'period':
        botao_point()
    if event.keysym == 'asterisk':
        botao_mult()
    if event.keysym == 'slash':
        botao_div()
    if event.keysym == 'plus':
        botao_som()
    if event.keysym == 'minus':
        botao_sub()
    if event.keysym == 'equal' or event.keysym == 'Return':
        botao_equ()
    if event.keysym == 'Delete':
        botao_c()
    if event.keysym == 'Escape':
        w.destroy()
    if event.keysym == 'm' or event.keysym == 'M':
        botao_mm()
    if event.keysym == 'BackSpace':
        botao_apaga()


w.bind('<Key>', key)
botao7 = Button(fr1A, activebackground='black',
                activeforeground='white', relief='flat', text=7, padx=20,
                pady=3, font='bold 50 bold', command=botao_7)
botao7.pack(side=LEFT)
botao8 = Button(fr1A, activebackground='black',
                activeforeground='white', relief='flat', text=8, padx=20,
                pady=3, font='bold 50 bold', command=botao_8)
botao8.pack(side=LEFT)
botao9 = Button(fr1A, activebackground='black',
                activeforeground='white', relief='flat', text=9, padx=20,
                pady=3, font='bold 50 bold', command=botao_9)
botao9.pack(side=LEFT)
botaoC = Button(fr, activebackground='black',
                activeforeground='red', relief='flat', text='C', padx=20,
                pady=3, font='bold 50 bold', command=botao_c, fg='red')
botaoC.pack(side=LEFT)
botaoMM = Button(fr, activebackground='black',
                 activeforeground='white', relief='flat', text='+/-', padx=20,
                 pady=3, font='bold 50 bold', command=botao_mm)
botaoMM.pack(side=RIGHT)
botao4 = Button(fr2A, activebackground='black',
                activeforeground='white', relief='flat', text=4, padx=20,
                pady=3, font='bold 50 bold', command=botao_4)
botao4.pack(side=LEFT)
botao5 = Button(fr2A, activebackground='black',
                activeforeground='white', relief='flat', text=5, padx=20,
                pady=3, font='bold 50 bold', command=botao_5)
botao5.pack(side=LEFT)
botao6 = Button(fr2A, activebackground='black',
                activeforeground='white', relief='flat', text=6, padx=20,
                pady=3, font='bold 50 bold', command=botao_6)
botao6.pack(side=LEFT)
botaoMult = Button(fr2B, activebackground='black',
                   activeforeground='white', relief='flat', text='x', padx=20,
                   pady=3, font='bold 50 bold', command=botao_mult)
botaoMult.pack(side=LEFT)
botao1 = Button(fr3A, activebackground='black',
                activeforeground='white', relief='flat', text=1, padx=20,
                pady=3, font='bold 50 bold', command=botao_1)
botao1.pack(side=LEFT)
botao2 = Button(fr3A, activebackground='black',
                activeforeground='white', relief='flat', text=2, padx=20,
                pady=3, font='bold 50 bold', command=botao_2)
botao2.pack(side=LEFT)
botao3 = Button(fr3A, activebackground='black',
                activeforeground='white', relief='flat', text=3, padx=20,
                pady=3, font='bold 50 bold', command=botao_3)
botao3.pack(side=LEFT)
botaoDiv = Button(fr3B, activebackground='black',
                  activeforeground='white', relief='flat', text='/', padx=25,
                  pady=3, font='bold 50 bold', command=botao_div)
botaoDiv.pack(side=LEFT)
botaoPoint = Button(fr4A, activebackground='black',
                    activeforeground='white', relief='flat', text='.', padx=30,
                    pady=3, font='bold 50 bold', command=botao_point)
botaoPoint.pack(side=LEFT)
botao0 = Button(fr4A, activebackground='black',
                activeforeground='white', relief='flat', text=0, padx=20,
                pady=3, font='bold 50 bold', command=botao_0)
botao0.pack(side=LEFT)
botaoSub = Button(fr4B, activebackground='black',
                  activeforeground='white', relief='flat', text='-', padx=30,
                  pady=3, font='bold 50 bold', command=botao_sub)
botaoSub.pack(side=LEFT)
botaoSom = Button(fr5B, activebackground='black',
                  activeforeground='white', relief='flat', text='+', padx=10,
                  pady=3, font='bold 50 bold',
                  command=botao_som)
botaoSom.pack(side=LEFT)
botaoEqu = Button(fr4A, activebackground='black',
                  activeforeground='white', relief='flat', text='=', padx=10,
                  pady=3, font='bold 50 bold', command=botao_equ)
botaoEqu.pack(side=LEFT)

w.mainloop()
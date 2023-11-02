#date: 2023-11-02T16:45:31Z
#url: https://api.github.com/gists/21ad24d1dd89e7f618f0fe59641d3fee
#owner: https://api.github.com/users/alladin3

import tkinter as tk
import matplotlib.pyplot as plt
from tkinter import *
from tkinter.messagebox import showerror, showinfo

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from tkinter import filedialog as fd, filedialog, messagebox


class Equation:
    def __init__(self, delta, A, omega):
        self.delta = delta
        self.A = A
        self.omega = omega

    def __call__(self, Y, t):
        theta, omega = Y
        dtheta_dt = omega
        domega_dt = -self.delta * omega - np.sin(theta) + self.A * np.sin(self.omega * t)
        return np.array([dtheta_dt, domega_dt])


class Solver:
    def __init__(self, f):
        self.f = f
        self.u0 = None
        self.tmin = None
        self.tmax = None
        self.step = None

    def params(self, u0=0, tmin=0, tmax=10, step=0.1):
        self.u0 = u0
        self.tmin = tmin
        self.tmax = tmax
        self.step = step

    def setfunc(self, f):
        self.f = f

    def rk4step(self, h: float, f, ui, ti):
        self.h = h
        k1 = h * f(ui, ti)
        k2 = h * f(ui + k1 / 2, ti + h / 2)
        k3 = h * f(ui + k2 / 2, ti + h / 2)
        k4 = h * f(ui + k3, ti + h)
        return ui + (k1 + 2 * (k2 + k3) + k4) / 6

    def odeint(self):
        N = int((self.tmax - self.tmin) / self.step) + 1
        self.u = np.zeros((N, 2))
        self.t = np.linspace(self.tmin, self.tmax, N)
        self.u[0] = self.u0
        for i in range(1, N):
            self.u[i] = self.rk4step(self.h, self.f, self.u[i - 1], self.t[i - 1])
        return self.u, self.t


class App:
    def __init__(self):
        self.equation = Equation(delta=0.1, A=1, omega=1)
        self.solver = Solver(self.equation)

        self.u0 = np.array([0, 0])
        self.tmin = 0
        self.tmax = 10
        self.step = 0.01

    def label_entry(self, frame,label_text):  # функция  создает виджеты  (Label)и(Entry) внутри родительского виджета (parent)
        label = tk.Label(frame, text=label_text)
        entry = tk.Entry(frame)
        return label, entry

    # Функция для создания кнопки
    def create_button(self, frame, text, command):
        button = tk.Button(frame, text=text, command=command)
        return button

    def create_root(
            self):  # if вызовешь кот.мурчало(), муркнут сразу все коты на свете if вызовешь self.мурчало(), муркнет только тот кот, на которого указывает self
        self.root = tk.Tk()  # Создаем главный объект ( окно приложения)
        self.root['bg'] = 'yellow'
        self.root.title('Строим графики')
        # Создаем рамки для группировки элементов
        self.frame1 = tk.Frame(self.root)
        self.frame2 = tk.Frame(self.root)
        self.frame3 = tk.Frame(self.root)
        self.frame4 = tk.Frame(self.root)
        self.frame5 = tk.Frame(self.root)
        self.frame6 = tk.Frame(self.root)
        self.frame7 = tk.Frame(self.root)
        self.frame8 = tk.Frame(self.root)
        self.frame9 = tk.Frame(self.root)
        self.frame10 = tk.Frame(self.root)

        # Создаем пары метка-ввод
        self.label1, self.entry1 = self.label_entry(self.frame1, "Enter potery:")
        self.label2, self.entry2 = self.label_entry(self.frame2, "Enter chastota:")
        self.label3, self.entry3 = self.label_entry(self.frame3, "Enter amplitude:")
        self.label4, self.entry4 = self.label_entry(self.frame4, "Enter tmax:")
        self.label7, self.entry7 = self.label_entry(self.frame7, "Enter tmin:")
        self.label8, self.entry8 = self.label_entry(self.frame8, "Enter step:")
        self.label6, self.entry6 = self.label_entry(self.frame6, "Entered parameters:")
        self.label9, self.entry9 = self.label_entry(self.frame9, "Enter U0:")
        self.label10, self.entry10 = self.label_entry(self.frame10, "Enter X0:")

        self.entry1.insert(0, "0.01")  # по умолчанию для вставки значения "x0" в виджет Entry (поле ввода текста)
        self.entry2.insert(0, "1.2")  # для вставки значения "dx" в виджет Entry (поле ввода текста)
        self.entry3.insert(0, "0.05")
        self.entry4.insert(0, "100")
        self.entry7.insert(0, "0.0")
        self.entry8.insert(0, "0.1")
        self.entry6.insert(0, "delta: A: omega: tmin: tmax: step: ")
        self.entry9.insert(0, "3.0")
        self.entry10.insert(0, "0.0")

        # Создание кнопок и их размещение
        # Frame(root)` создает фрейм `button_frame` внутри родительского виджета `root`. Это означает, что `button_frame` будет отображаться внутри `root` и наследовать его свойства и методы
        self.button_frame = tk.Frame(
            self.root)  # button_frame`  использован для группировки кнопок или других виджетов, связанных с кнопками, в одном блоке
        self.button1 = self.create_button(self.frame5, "Задать параметры", self.set_parameters)
        self.button2 = self.create_button(self.frame5, "Проверить параметры", self.check_parameters)
        self.button3 = self.create_button(self.frame5, "Вычислить", self.calculate_table)
        self.button4 = self.create_button(self.frame5, "Coxpанить ", self.write_table_to_file)
        self.button5 = self.create_button(self.frame5, "Считать таблицу значений из файла", self.read_table_from_file)
        self.button6 = self.create_button(self.frame5, "Нарисовать график", self.draw_graph)
        self.button7 = self.create_button(self.frame5, "Сохранить график", self.save_graph)
        self.button8 = self.create_button(self.frame5, "Очистить график", self.clear_graph)
        self.button9 = self.create_button(self.frame5, "Закрыть приложение", self.close_application)

        self.table = tk.Text(self.root, width=20, height=15, wrap=tk.NONE)
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.navigation = NavigationToolbar2Tk(self.canvas, self.root)

        # Размещаем все элементы на окне
        self.frame1.pack(anchor=NW, side=TOP, pady=5)
        self.label1.pack(side=LEFT, padx=5, pady=5)
        self.entry1.pack(side=RIGHT, padx=5, pady=5)

        self.frame2.pack(anchor=NW, side=TOP, pady=5)
        self.label2.pack(side=LEFT, padx=5, pady=5)
        self.entry2.pack(side=RIGHT, padx=5, pady=5)

        self.frame3.pack(anchor=NW, side=TOP, pady=5)
        self.label3.pack(side=LEFT, padx=5, pady=5)
        self.entry3.pack(side=RIGHT, padx=5, pady=5)

        self.frame4.pack(anchor=NW, side=TOP, pady=5)
        self.label4.pack(side=LEFT, padx=5, pady=5)
        self.entry4.pack(side=RIGHT, padx=5, pady=5)

        self.frame6.pack(anchor=NW, side=TOP, pady=5)
        self.label6.pack(side=LEFT, padx=5, pady=5)
        self.entry6.pack(side=RIGHT, padx=5, pady=5)

        self.frame7.pack(anchor=NW, side=TOP, pady=5)
        self.label7.pack(side=LEFT, padx=5, pady=5)
        self.entry7.pack(side=RIGHT, padx=5, pady=5)

        self.frame8.pack(anchor=NW, side=TOP, pady=5)
        self.label8.pack(side=LEFT, padx=5, pady=5)
        self.entry8.pack(side=RIGHT, padx=5, pady=5)

        self.frame9.pack(anchor=N, side=LEFT, pady=5)
        self.label9.pack(side=TOP, padx=1, pady=5)
        self.entry9.pack(side=BOTTOM, padx=1, pady=5)

        self.frame10.pack(anchor=N, side=LEFT, pady=5)
        self.label10.pack(side=TOP, padx=1, pady=5)
        self.entry10.pack(side=BOTTOM, padx=1, pady=5)

        self.ax = self.fig.add_subplot(111)  # новая область рисования графика


        self.frame5.pack(anchor=E,side=TOP)
        self.table.pack(anchor=NW, side=tk.LEFT)
        self.button1.pack(anchor=NW, side=tk.LEFT)
        self.button2.pack(anchor=NW, side=tk.LEFT)
        self.button3.pack(anchor=NW, side=tk.LEFT)
        self.button4.pack(anchor=NW, side=tk.LEFT)
        self.button5.pack(anchor=NW, side=tk.LEFT)
        self.button6.pack(anchor=NW, side=tk.LEFT)
        self.button7.pack(anchor=NW, side=tk.LEFT)
        self.button8.pack(anchor=NW, side=tk.LEFT)
        self.button9.pack(anchor=NW, side=tk.LEFT)

        self.root.mainloop()  # Запускаем постоянный цикл, чтобы программа работала

    # def evaluate(self, func_str: str, x):
    #  return eval(func_str.replace('x', x))

    # open button

    def set_parameters(self, tmin=0, u0=np.array([3.0, 0]), tmax=100, step=0.1):
        # app.set_equation_params(delta=0.1, A=1, omega=1)
        self.u0 = u0
        self.tmin = tmin
        self.tmax = tmax
        self.step = step
        self.delta = self.entry1.get()
        self.omega = self.entry2.get()
        self.A = self.entry3.get()
        self.tmax = self.entry4.get()
        self.tmin = self.entry7.get()
        self.step = self.entry8.get()
        self.u0 = self.entry9.get()
        self.x0 = self.entry10.get()

        self.entry6.delete(0, END)
        # insert принимает индекспозиции и строку для вставки.
        self.entry6.insert(0, str(self.delta) + ' ' + str(self.omega) + ' ' + str(self.A) + ' ' + str(
            self.tmax) + '' + str(self.tmin) + '' + str(step) + '')
        print(self.delta, self.omega, self.A, self.tmin, self.tmax, self.step, sep='|')

    def check_parameters(self):
        try:
            self.delta = float(self.delta)
        except:
            self.error()
        try:
            self.omega = float(self.omega)
        except:
            self.error()
        try:
            self.A = float(self.A)
        except:
            self.error()
        try:
            self.tmin = float(self.tmin)
        except:
            self.error()
        try:
            self.tmax = float(self.tmax)
        except:
            self.error()
        try:
            self.step = float(self.step)
        except:
            self.error()
        # Функция, вызываемая при нажатии кнопки "Проверить параметры"

    # Функция, вызываемая при нажатии кнопки "Рассчитать таблицу значений"
    def calculate_table(self):
        N = int((self.tmax - self.tmin) / self.step) + 1
        u = np.zeros((N, 2))
        t = np.linspace(self.tmin, self.tmax, N)
        u[0] = self.u0
        for i in range(1, N):
            u[i] = self.solver.rk4step(0.01, self.solver.f, u[i - 1], t[i - 1])

        self.table.delete("1.0", END)  # с позиции строки 1 и позиции столбца 0 бедут до конца удален
        text = ''  # После удаления текста, переменная text будет содержать пустую строку ('').
        for i in range(0, N):
            text += f"{round(u[i][0], 3)} {round(u[i][1], 3)} {round(t[i], 3)}\n"  # добавляет текстовую строку в виджет str(self.table[i])
        self.table.insert(tk.END, text)
        print(text)
        self.u = u
        self.t = t
        return u, t

    def select_file(self):
        filetypes = (('text files', '*.txt'), ('All files',
                                               '*.*'))  # создается кортеж filetypes определяет типы файлов, которые будут отображаться в диалоговом окне.
        filename = fd.askopenfilename(title='Open a file', initialdir='/',
                                      filetypes=filetypes)  # начальная директория для открытия диалогового окна типы файлов, указанные в кортеже `filetypes`
        showinfo(title='Selected File', message=filename)

    def write_table_to_file(self):
        N = int((self.tmax - self.tmin) / self.step) + 1
        file_path = filedialog.asksaveasfilename(defaultextension=".txt")
        # Проверяем, был ли выбран файл
        if file_path:
            # Открываем файл для записи
            with open(file_path, "w") as f:
                for i in range(1, N):  # возвращает количество элементов в списке
                    f.write(str(self.t[i]) + ' ')  # в строковый формат перед их записью в файл.
                    for j in range(0, len(self.u[i])):  # возвращает количество элементов в списке
                        f.write(str(self.u[i][j]) + ' ')  # в
                    f.write(' \n')
        self.table.delete("1.0", END)
        text = ''
        for i in range(0, N):  # добавляет содержимое двух списков  в виде строк в конец виджета таблицы
            text += f"{round(self.u[i][0], 3)} {round(self.u[i][1], 3)} {round(self.t[i], 3)}\n"
        self.table.insert(tk.END, text)

    def read_table_from_file(self):
        file_path = filedialog.askopenfilename(title="Выберите файл", filetypes=[("Текстовые файлы", "*.txt")])
        if file_path:
            try:
                with open(file_path, 'r') as file:
                    self.table_delta = []  # создает пустые списки self.table_x и self.table_f.
                    self.table_omega = []
                    self.table_f = []
                    for i in file.readlines():  # перебирает каждую строку файла,
                        self.table_delta.append(float(i.split(' ')[0]))  # строка разбивается на элементы по пробелу с помощью метода split()
                        self.table_omega.append(float(i.split(' ')[1]))
                        self.table_f.append(float(i.split(' ')[ 2]))  # Первый элемент после разбиения преобразуется в тип float и добавляется в список self.table_x, а второй элемент - в список self.table_f.
                    file.seek(0)
                    self.table.insert(tk.END, file.read())
                    return 0
            except FileNotFoundError:
                messagebox.showinfo('', 'нет такого файла')
                text = self.table.get("1.0", END).split('\n')
                self.table_delta = np.full((self.tmax, 1), fill_value=np.nan)
                self.table_omega = np.full((self.tmax, 1), fill_value=np.nan)
                self.table_f = np.full((self.tmax, 1), fill_value=np.nan)
                for i in range(1,
                               len(text)):  # со второго элемента  строка разбивается на элементы по пробелу с помощью метода split(' ').
                    if len(text[i].split(' ')) == 3:  # проверяется, что разбитая строка содержит ровно 2 элемента
                        self.table_delta[i - 1] = float(
                            text[i].split(' ')[0])  # меньшается на 1 (i - 1), так как индексация начинается с 0.
                        self.table_omega[i - 1] = float(text[i].split(' ')[1])
                        self.table_f[i - 1] = float(text[i].split(' ')[2])


        # Функция, вызываемая при нажатии кнопки "Считать таблицу значений из файла"

    def draw_graph(self):
        self.ax.plot(self.t, self.u[:, 1], self.tmin, self.tmax, self.step)
        # `Fig` - контейнер, в котором будут располагаться все элементы графика
        # Ax этообластьграфика, на которой будут отображаться данные
        # plt.plot(self.table_x, self.table_f, figure=self.fig)
        self.solver.params(self.u0, self.tmin, self.tmax, self.step)
        u, t = self.solver.odeint()
        theta = u[:, 0]
        plt.plot(t, theta)
        plt.xlabel('Time')
        plt.ylabel('Theta')
        plt.title('Pendulum Motion')
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()  # Разместить холст в контейнере с помощью

    def error(self):
        showerror(title='Ошибка!', message='Please,enter another type of number')

    def save_graph(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".png")
        self.fig.savefig(file_path)
        # Функция, вызываемая при нажатии кнопки "Сохранить график"

    def clear_graph(self):
        self.ax.clear()  # очищает график от кривых, сам график остается
        #       ax.cla()   так тоже работает
        self.canvas.draw()  # обновляет холст, без этого изменений видно не будет
        # for item in self.canvas.get_tk_widget().find_all():
        # self.canvas.get_tk_widget().delete(item)

    def close_application(self):
        self.root.destroy()
        # Функция, вызываемая при нажатии кнопки "Закрыть приложение"


# def evaluate(func_str: str, x): #пределяет выражение для вычисления.заменяет символ 'x' в строке функции на значение x
#  return eval(func_str.replace('x', str(x)))# и затем использует функцию eval() для вычисления значения этой строки в качестве выражения.


app = App()  # app-экземпляр класса
app.create_root()

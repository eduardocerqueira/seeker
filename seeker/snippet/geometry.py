#date: 2022-02-21T17:05:34Z
#url: https://api.github.com/gists/7caf89512ef8f5bea01015b3f2337da6
#owner: https://api.github.com/users/ap75

"""
Создаем библиотеку классов для следующих графических объектов:
- точка,
- прямая,
- отрезок,
- треугольник,
- прямоугольник, стороны которого параллельны осям координат,
- круг,
- ломаная линия (задание для Игоря, Льва и Максима).

В каждом классе должны быть реализованы (свои или унаследованные) конструктор и методы для вычисления:
- площади объекта, функция называется p() (названия сами придумывали, я не виноват ☺),
- периметра объекта, функция s(),
- определения принадлежности заданной точки объекту, функция have(point) (только для первой группы),
- строкового представления объекта, __str__() (только для второй группы).

В конструкторе объекта необходимо проверить правильность заданных свойств объекта (задание
только для второй группы), например, задание двух РАЗНЫХ точек в качестве свойств прямой и
в случае неверно заданных свойств выбросить исключение (выполнить команду raise).

Частично выполненное задание приведено ниже:
"""

import math


class Point:
  '''
  Объект "точка"
  '''
  def __init__(self, x, y):
    self.x, self.y = x, y

  def __str__(self):
    return f"Точка ({self.x},{self.y})"

  def p(self):
    # Считаем площадь точки равной 0
    return 0

  def s(self):
    # Считаем периметр точки равным 0
    return 0

  def have(self, point):
    return self.x == point.x and self.y == point.y


class Line(Point):
  '''
  Объект "прямая"
  '''
  def __init__(self, point1, point2):
    if point1.have(point2):
      raise Exception('Неверные свойства объекта: координаты точек совпадают')
    self.point1 = point1
    self.point2 = point2

  def __str__(self):
    return f"Прямая ({self.point1}-{self.point2})"

  def p(self):
    # Считаем периметр прямой равным бесконечности
    return float('inf')

  # Метод s(self): переопределять не нужно, считаем площадь прямой равной 0
  # Поэтому здесь будет использован метод, унаследованный от класса Point

  def have(self, point):
    # Проверяем, что точка принадлежит заданной прямой
    # Для этого вычисляем коэффициенты уравнения прямой вида y=kx+c
    k = (self.point1.y - self.point2.y) / (self.point1.x - self.point2.x)
    c = self.point1.y - self.point1.x * k
    return point.x * k + c == point.y


class Length(Line):
  '''
  Объект "отрезок"
  '''

  # Методы __init__() и s() не переопределяем, а наследуем

  def __str__(self):
    return f"Отрезок ({self.point1}-{self.point2})"

  def p(self):
    return math.sqrt(
      (self.point1.y - self.point2.y) ** 2 +
      (self.point1.x - self.point2.x) ** 2)

  def have(self, point):
    return (
      # Проверяем, что точка принадлежит прямой, частью которой является отрезок
      super().have(point) and
      # и хотя бы одна из координат находится внутри заданного
      # диапазона, т.е. принадлежит отрезку
      (self.point1.x <= point.x <= self.point2.x or
      self.point1.x >= point.x >= self.point2.x))


class Triangle(Length):
  '''
  Объект "треугольник"
  '''
  def __init__(self, point1, point, point3):
    pass

# и т.д. ...

point1 = Point(10,10)
Length(point1, Point(10,11)) # Порядок, прямая задана корректно
Length(point1, Point(10,10)) # Здесь должно произойти исключение

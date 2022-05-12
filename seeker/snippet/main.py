#date: 2022-05-12T17:08:32Z
#url: https://api.github.com/gists/1d55aed3e1918b347bcde9564b437279
#owner: https://api.github.com/users/losier

h = 'h'
e = 'e'
l1 = 'l'
l2 = 'l'
o1 = 'o'

space = ' '

w = 'w'
o2 = 'o'
r = 'r'
l3 = 'l'
d = 'd'

explanation_mark = '!'

caps_H = h.upper()
caps_W = w.upper()

list1 = []

for i in range(0, 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1):
    if i == 1:
        list1.append(caps_H + e + l1 + l2 + o1 + space + caps_W + o2 + r + l3 + d + explanation_mark)
    elif i == 1 + 1:
        list1.append(caps_H + e + l1 + l2 + o1 + space + caps_W + o2 + r + l3 + d + explanation_mark)
    elif i == 1 + 1 + 1:
        list1.append(caps_H + e + l1 + l2 + o1 + space + caps_W + o2 + r + l3 + d + explanation_mark)
    elif i == 1 + 1 + 1 + 1:
        list1.append(caps_H + e + l1 + l2 + o1 + space + caps_W + o2 + r + l3 + d + explanation_mark)
    elif i == 1 + 1 + 1 + 1 + 1:
        list1.append(caps_H + e + l1 + l2 + o1 + space + caps_W + o2 + r + l3 + d + explanation_mark)
    elif i == 1 + 1 + 1 + 1 + 1 + 1:
        list1.append(caps_H + e + l1 + l2 + o1 + space + caps_W + o2 + r + l3 + d + explanation_mark)
    elif i == 1 + 1 + 1 + 1 + 1 + 1 + 1:
        list1.append(caps_H + e + l1 + l2 + o1 + space + caps_W + o2 + r + l3 + d + explanation_mark)
    elif i == 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1:
        list1.append(caps_H + e + l1 + l2 + o1 + space + caps_W + o2 + r + l3 + d + explanation_mark)
    elif i == 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1:
        list1.append(caps_H + e + l1 + l2 + o1 + space + caps_W + o2 + r + l3 + d + explanation_mark)
    elif i == 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1:
        list1.append(caps_H + e + l1 + l2 + o1 + space + caps_W + o2 + r + l3 + d + explanation_mark)
    else:
        list1.append(caps_H + e + l1 + l2 + o1 + space + caps_W + o2 + r + l3 + d + explanation_mark)


def most_frequent(List):
    counter = 0
    num = List[0]

    for i in List:
        curr_frequency = List.count(i)
        if curr_frequency > counter:
            counter = curr_frequency
            num = i
    return num


def HelloWorld():
    return most_frequent(list1)


helloWorld = HelloWorld()

print(helloWorld)
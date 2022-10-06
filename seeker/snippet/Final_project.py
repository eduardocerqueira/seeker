#date: 2022-10-06T17:28:39Z
#url: https://api.github.com/gists/6d56ee90d610130ddef56f5c6dcddfdd
#owner: https://api.github.com/users/mohmmadmasri

class bus:
    def __init__(self, num, start, end, free_sit):
        self.__num = num
        self.__start = start
        self.__end = end
        self.__sits = 35
        self.__free_sit = free_sit

    def get_num(self):
        return self.__num

    def get_start(self):
        return self.__start

    def get_end(self):
        return self.__end

    def get_free_sits(self):
        return self.__free_sit

    def set_free_sits(self, free_sit):
        self.__free_sit = free_sit


obj1 = bus("1", "GAZA", "NABLUS", 35)
obj2 = bus("2", "RAFAH", "NABLUS", 35)
obj3 = bus("3", "GAZA", "RAFAH", 35)

print("number of bus=", obj1.get_num(), "\n" "start", obj1.get_start(), "\n""end", obj1.get_end(), "\n""free stis =",
      obj1.get_free_sits())
print("************")
print("number of bus=", obj2.get_num(), "\n" "start", obj2.get_start(), "\n""end", obj2.get_end(), "\n""free sits =",
      obj2.get_free_sits())
print("************")
print("number of bus=", obj3.get_num(), "\n" "start", obj3.get_start(), "\n""end", obj3.get_end(), "\n""free sits =",
      obj3.get_free_sits())
print("************")
position1 = 0
position2 = 0
position3 = 0
while 1:

    x = int(input("enter number of bus"))
    if (x == 1):
        print("number of free sti =", obj1.get_free_sits())
        if (obj1.get_free_sits() > 0):
            z = int(input("enter 1 to book or 0 to exit"))
            if (z == 0):
                break
            elif (z == 1):
                obj1.set_free_sits(obj1.get_free_sits() - 1)
                position1 += 1
                if (position1 <= 10):
                    print('Your chair is in the front.')
                elif position1 > 10 and position1 <= 20:
                    print('Your chair is in the middle.')
                elif position1 > 20 and position1 <= 35:
                    print('Your chair is in the end.')
        else:
            print('No more freesits!')
    if (x == 2):
        print("number of free sti =", obj2.get_free_sits())
        if (obj2.get_free_sits() > 0):
            z = int(input("enter 1 to book or 0 to exit"))
            if (z == 0):
                break
            elif (z == 1):
                obj2.set_free_sits(obj2.get_free_sits() - 1)
                position2 += 1
                if (position2 <= 10):
                    print('Your chair is in the front.')
                elif (position2 > 10 and position2 <= 20):
                    print('Your chair is in the middle.')
                elif (position2 > 20 and position2 <= 35):
                    print('Your chair is in the end.')
        else:
            print('No more free sits!')
    if (x == 3):
        print("number of free sits =", obj3.get_free_sits())
        if (obj3.get_free_sits() > 0):
            z = int(input("enter 1 to book or 0 to exit"))
            if (z == 0):
                break
            elif (z == 1):
                obj3.set_free_sits(obj3.get_free_sits() - 1)
                position3 += 1
                if (position3 <= 10):
                    print('Your chair is in the front.')
                elif (position3 > 10 and position3 <= 20):
                    print('Your chair is in the middle.')
                elif (position3 > 20 and position3 <= 35):
                    print('Your chair is in the end.')

        else:
            print('No more free sits!')

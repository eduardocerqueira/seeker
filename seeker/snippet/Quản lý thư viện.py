#date: 2022-10-24T17:33:21Z
#url: https://api.github.com/gists/7ac7852b10e67cfd1ed78e4c18a5c45a
#owner: https://api.github.com/users/Khanhduong123

"Code linkedlist"
import sys
class Book:
    def __init__(self,code,title,author,price):
        self.code = code
        self.title = title
        self.author = author
        self.price = price


    def print(self):
        print("Code: ",self.code)
        print("Title: ", self.title)
        print("Author: ", self.author)
        print("Price: ", self.price)


class Node:
    def __init__(self,data):
        self.data = data
        self.next = None

class BookList:
    def __init__(self):
        self.__head = None

    def append(self,data):
        newNode = Node(data)
        if self.__head is None:
            self.__head = newNode
        else:
            current = self.__head
            while current.next is not None:
                current = current.next
            current.next = newNode


    def duplicate(self,code):
        current = self.__head
        while current is not None:
            if current.data.code == code:
                return True
            current = current.next
        return False

    def search(self,title):
        current = self.__head
        while current is not None:
            if current.data.title == title:
                current.data.print()
            current = current.next
        return False

    def delete(self,code):
        current = self.__head
        prev = None
        while current is not None:
            if current.data.code==code:
                prev.next = current.next
            prev = current
            current=current.next

    def display(self):
        current = self.__head
        count = 0
        while current is not None:
            print("\nBook",count,"info's: ")
            current.data.print()
            current = current.next
            count +=1

    def update(self,code,option):
        current = self.__head
        while current is not None:
            if option == 1 and current.data.code == code:
                print("Input the new title:",end=" ")
                title = input()
                current.data.title = title

            if option == 2 and current.data.code == code:
                print("Input the new author:",end=" ")
                author = input()
                current.data.author = author

            if option == 3 and current.data.code == code:
                print("Input the new price:",end=" ")
                price = int(input())
                current.data.price = price
            print("Update successfully")
            current = current.next


#coding menu
menu ={1:"Add new book",
       2:"Search a book by title",
       3:"Update a book",
       4:"Delete a book",
       5:"Display all book",
       6:"Exit"}

def printMenu():
    print("========Menu========")
    for key in menu:
        print("==",key,":",menu[key])
    print("====================")

def add_book(l:BookList):
    code = input("Input code: ")
    title = input("Input title: ")
    author = input("Input author: ")
    price = int(input("Input price: "))
    newbook = Book(code,title,author,price)
    "Check duplicate from code of book"

    if not l.duplicate(code):
        l.append(newbook)
        print("Book added successfully")
    else:
        print("Book's code is duplicated")

def search_book(l:BookList):
    print("Input title you want to search:",end=" ")
    title = input()
    return l.search(title)

def update_book(l:BookList):

    menu_option={1:"Update the title",
                 2:"Update the author",
                 3:"Update the price"}

    for key in menu_option:
        print(key, menu_option[key])
    print("Choose the option you want to update:",end=" ")
    option = int(input())
    print("Input the code of book you want to update:", end=" ")
    code = input()
    if option == 1:
        l.update(code,option)
    elif option == 2:
        l.update(code,option)
    elif option == 3:
        l.update(code,option)

def delete(l:BookList):
    print("Input code you want to delete:",end=" ")
    code = input()
    return l.delete(code)

def display(l:BookList):
    return l.display()

def exist():
    print("Thank you, Good bye Baby!")
    sys.exit()


if __name__ == "__main__":
    bk= BookList()
    while True:
        printMenu()
        print("Choose the option you want to do:",end=" ")
        user_choose = int(input())
        if user_choose == 1:
            add_book(bk)
        elif user_choose == 2:
            search_book(bk)
        elif user_choose == 3:
            update_book(bk)
        elif user_choose == 4:
            delete(bk)
        elif user_choose == 5:
            display(bk)
        elif user_choose ==6 :
            exist()
        else:
            print("Invalid Number")


"Stack"





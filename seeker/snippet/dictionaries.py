#date: 2023-10-27T16:46:15Z
#url: https://api.github.com/gists/f34aa78050aeddc8a82470375913b6ba
#owner: https://api.github.com/users/chathudilzo



def main():
    inventory={}

    value=100
    print("INVENTORY MANAGEMENT SYSTEM!")
    while value>0:
        print(inventory)

        print("press 1 to add new item")
        print("Press 2 to update an item")
        print("press 3 to remove an item")
        print("press 4 to check quantity")
        print("press 5 to calculate total stock value")
        print("press 0 to EXIT")

        value =input("Choose one from (0,1,2,3,4,5): ")

        try:
            value=int(value)
            if value==0:
                print("Exiting the system")
            elif value==1:
                item=input("Enter item name: ").lower()
                price=float(input("Enter price: "))
                quantity=int(input("Enter quantity: "))
                if item==None or price==0 or quantity==0:
                    print("Cannot have empty values")
                else:
                    add_item(inventory,item,price,quantity)
            elif value==2:
                item=input("Enter item name: ").lower()
                price=float(input("Enter price: "))
                quantity=int(input("Enter quantity: "))
                if item==None or price==0 or quantity==0:
                    print("Cannot have empty values")
                else:
                    update_item(inventory,item,price,quantity)
            elif value==3:
                item=input("Enter item name: ")

                remove_item(inventory,item)
            elif value==4:
                item=input("Item name of quantity check: ")
                check_quantity(inventory,item)
            elif value==5:
                total=calculate_inventory_value(inventory)
                print(f"Total Value:{total}")
            else:
                print("Invalid input!")
        except ValueError:
            print("You didnt entered a Integer!")
            value=100



def update_item(inventory,item,price,quantity):
    if item in inventory:
        inventory[item]=(price,quantity)
    else:
        print("Item not found!")

def add_item(inventory,item,price,quantity):
    if item in inventory:
        price,current_quantity=inventory[item]
        inventory[item]=(price,current_quantity+quantity)
    else:
        inventory[item]=(price,quantity)

def remove_item(inventory,item):
    if item in inventory:
        del inventory[item]
    else:
        print("Item not found!")

def check_quantity(inventory,item):
    if item in inventory:
        price,quantity=inventory[item]

        print(quantity)


def calculate_inventory_value(inventory):

    total_value=sum(price*quantity for price,quantity in inventory.values())

    return total_value





if __name__=='__main__':
    main()

#date: 2022-10-05T17:25:11Z
#url: https://api.github.com/gists/fef161678ca9acc31041199ac70ad048
#owner: https://api.github.com/users/Pymmdrza

import random
from rich import print as printx
import bitcoin
import random
import multiprocessing


def Pros():
    z = 0
    fou = 0
    target: str = "1Mmdrza"
    target2: str = "1mmdrza"
    printx(f"""
    [gold1]╔════════════════════════════════════════════════════════════════════════════════════════╗                              [/gold1]
    [gold1]║              [red1]╔═╗╦═╗╔═╗╔═╗╦═╗╔═╗╔╦╗╔╦╗╔═╗╦═╗[/red1][grey66]  ╔╦╗╔╦╗╔╦╗╦═╗╔═╗╔═╗ ╔═╗╔═╗╔╦╗[/grey66]              ║[/gold1]
    [gold1]║              [red3]╠═╝╠╦╝║ ║║ ╦╠╦╝╠═╣║║║║║║║╣ ╠╦╝[/red3][grey99]  ║║║║║║ ║║╠╦╝╔═╝╠═╣ ║  ║ ║║║║[/grey99]              ║[/gold1]
    [gold1]║              [red1]╩  ╩╚═╚═╝╚═╝╩╚═╩ ╩╩ ╩╩ ╩╚═╝╩╚═[/red1][grey66]  ╩ ╩╩ ╩═╩╝╩╚═╚═╝╩ ╩o╚═╝╚═╝╩ ╩[/grey66]              ║[/gold1]
    [gold1]╚════════════════════════════════════════════════════════════════════════════════════════╝                              [/gold1]
    [red][[white]+[/white]] Target 1  : [/red][green1]{target}   [/green1]
    [red][[white]+[/white]] Target 2  : [/red][green1]{target2}  [/green1]
    [red][[white]+[/white]] Character : [/red][green1]First [0:6][/green1]
    [red][[white]+[/white]] Address Generator 1 :[/red][gold1] Random from :[cyan] 80000000003333333333333333333333333333000000[/cyan][/gold1]
    [red][[white]+[/white]] Address Generator 1 :[/red][gold1] Random To : [cyan]333333333333333000000033333333333333333333333333333333333364[/cyan][/gold1]
    [red][[white]+[/white]] Address Generator 2 :[/red][gold1] Dec From Counter 1 to ....[/gold1]
    [red][[white]+[/white]][/red][magenta] Module Bitcoin And MultiProcess Active [/magenta]
    [red3]==========================================================================================[/red3]
        """)
    while True:
        z += 1
        dec = int(random.randint(80000000003333333333333333333333333333000000,
                                 333333333333333000000033333333333333333333333333333333333364))
        keyx = "%064x" % dec
        key2 = "%064x" % z
        private_key2: str = key2
        private_key: str = keyx
        address2 = bitcoin.privkey_to_address(private_key2)

        address = bitcoin.privkey_to_address(private_key)

        faddr = str(address)[0:6]
        faddr2 = str(address2)[0:6]

        if str(faddr) == str(target) or str(faddr) == str(target2):
            fou += 1
            with open("FoundVanity2.txt", "a") as fx:
                fx.write(
                    f"{address}\n{private_key}\n------------------------------------\n{address2}\n{private_key2}\n----------------------------\n")
                fx.close()
            print(z, address)
            print(z, private_key)
            print(z, address2)
            print(z, private_key2)

        elif str(faddr2) == str(target) or str(faddr2) == str(target2):
            fou += 1
            with open("FoundVanity2.txt", "a") as f1x:
                f1x.write(
                    f"{address}\n{private_key}\n------------------------------------\n{address2}\n{private_key2}\n----------------------------\n")
                f1x.close()
            print(z, address)
            print(z, private_key)
            print(z, address2)
            print(z, private_key2)

        else:
            print(f"[Total: {z}][FOUND: {fou}][{address}][{address2}]", end="\r")


if __name__ == '__main__':
    target = multiprocessing.Process(target=Pros)
    target.start()
    target.join()

#date: 2022-06-02T17:16:20Z
#url: https://api.github.com/gists/e57b1722cfa7b22e35fcba46499cf77d
#owner: https://api.github.com/users/sebastianet

tinc una peça de codi en python que envia un texte a un BOT de Telegram : client.py

Ara es crida .. des CRONTAB via un SH :

===
    szTG="($szTs) $HOSTNAME ($szWho) $eIP $szNet"
    /home/pi/python/telegram/client.py  "$szTG"
===

Aixo no es pas part de cap projecte, ni es un modul - es pot fer servir des de molts llocs.

Com faig per cridar-lo des el modul que dibuixa grafiques com el teu "piheat" ?

===
/home/pi/python/pkw/bin/pkw_cli.py  
    ---> /home/pi/python/pkw/bin/pkw_pkg/pkw_reader.py 
        ---> aqui hi pot haver error de conexio
            ---> /home/pi/python/telegram/client.py
===
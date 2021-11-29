#date: 2021-11-29T17:10:19Z
#url: https://api.github.com/gists/cb1896e6fad269286e902d5a4a2c0f25
#owner: https://api.github.com/users/pigmonchu


     +----------------+              +-----------------+
     |  FABRICANTES   |              |     ARTICULOS   |
     +----------------+ 1..1         +-----------------+
     | id: int (pKey) |-------+      | id: int (pKey)  |
     | nombre: Text   |       |      | nombre: Text    |
     +----------------+       |      | precio: Real    |
                              +------| fabricante: int |
                                0..n +-----------------+
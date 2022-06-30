#date: 2022-06-30T21:32:33Z
#url: https://api.github.com/gists/d409769532960918ceb35d241e64ae46
#owner: https://api.github.com/users/renefb

for competicao in numero_partidas.keys():

    for temporada in range(2022, 2023):
        
        url_competicao = f'https://www.cbf.com.br/futebol-brasileiro/competicoes/{competicao}/{temporada}'
        res = req.get(url_competicao)
        soup = BeautifulSoup(res.text)

        ids_partida = []
        # usando a função auxiliar de filtro:
        for a in soup.find_all(href=filtra_links_partidas):
            href = a['href']
            id_partida = href.split('/')[-1].split('?')[0]
            ids_partida.append(int(id_partida))

        numero_partidas[competicao][temporada] = max(ids_partida)
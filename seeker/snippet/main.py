#date: 2025-08-11T17:04:47Z
#url: https://api.github.com/gists/ac9a26c23920b3412921a365ba27026f
#owner: https://api.github.com/users/docafavarato

import requests
import networkx as nx
import matplotlib.pyplot as plt
import base64
import time
import json
from pyvis.network import Network

with open("data.json") as json_data: # Esse arquivo guarda os clients da API do Spotify, se quiser rodar o código vai ter
    d = json.load(json_data)		 # que criar um app na parte de devs do spotify e pegar lá.
    json_data.close()

CLIENT_ID = d["CLIENT_ID"]
CLIENT_SECRET = "**********"

 "**********"d "**********"e "**********"f "**********"  "**********"g "**********"e "**********"t "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"( "**********") "**********": "**********"
    auth_string = f"{CLIENT_ID}: "**********"
    b64_auth_string = base64.b64encode(auth_string.encode()).decode()

    headers = {
        "Authorization": f"Basic {b64_auth_string}",
        "Content-Type": "application/x-www-form-urlencoded"
    }

    data = {"grant_type": "client_credentials"}

    res = requests.post("https: "**********"
    if res.status_code != 200:
        raise Exception(f"Erro ao obter token: "**********"

    return res.json()["access_token"]

TOKEN = "**********"
HEADERS = {
    "Authorization": "**********"
}

from pyvis.network import Network

def plot_interactive_graph(G, first_artist, second_artist, start_year, end_year):
    net = Network(height='750px', width='100%', bgcolor='#222222', font_color='white')
    net.barnes_hut()

    for node in G.nodes:
        color = "skyblue"
        if first_artist.lower() in node.lower():
            color = "orange"
        elif second_artist.lower() in node.lower():
            color = "red"
        net.add_node(node, label=node, color=color)

    for edge in G.edges:
        net.add_edge(edge[0], edge[1])

    title = f"Grafo de Colaborações: {first_artist} & {second_artist} ({start_year}-{end_year})"
    net.heading = title
    net.set_options("""
        {
        "nodes": {
            "shape": "dot",
            "size": 16,
            "font": { "size": 16, "color": "#ffffff" }
        },
        "edges": {
            "color": "#cccccc",
            "width": 1.5
        },
        "physics": {
            "barnesHut": {
            "gravitationalConstant": -8000,
            "centralGravity": 0.3,
            "springLength": 150
            },
            "stabilization": {
            "iterations": 2500
            }
        }
        }
        """)

    net.write_html(f"interactive_{first_artist}-{second_artist}-{start_year}-{end_year}.html")
    print(f"Grafo salvo como 'interactive_{first_artist}-{second_artist}-{start_year}-{end_year}.html'. Abra no navegador.")

def get_artist_info(artist_name):
    """Retorna ID e nome correto do artista."""
    url = "https://api.spotify.com/v1/search"
    params = {"q": artist_name, "type": "artist", "limit": 1}
    res = requests.get(url, headers=HEADERS, params=params)
    if res.status_code != 200:
        raise Exception(f"Erro ao buscar artista {artist_name}: {res.text}")
    items = res.json()["artists"]["items"]
    if not items:
        raise Exception(f"Artista '{artist_name}' não encontrado.")
    artist = items[0]
    return artist["id"], artist["name"]

def get_collaborations_by_period(artist_id, artist_name, start_year, end_year):
    base_url = f"https://api.spotify.com/v1/artists/{artist_id}/albums"
    params = {"include_groups": "album,single,appears_on", "limit": 50}
    all_albums = []
    next_url = base_url

    while next_url:
        res = requests.get(next_url, headers=HEADERS, params=params)
        if res.status_code != 200:
            print(f"Erro ao obter álbuns de {artist_name}: {res.text}")
            break
        data = res.json()
        all_albums.extend(data.get("items", []))
        next_url = data.get("next")
        params = None

    results = set()
    for album in all_albums:
        release_date = album.get("release_date", "")
        if not release_date or len(release_date) < 4:
            continue
        release_year = int(release_date[:4])
        if not (start_year <= release_year <= end_year):
            continue

        album_id = album.get("id")
        if not album_id:
            continue

        try:
            tracks_res = requests.get(f"https://api.spotify.com/v1/albums/{album_id}/tracks", headers=HEADERS)
            if tracks_res.status_code != 200:
                continue

            tracks = tracks_res.json().get("items", [])
            for track in tracks:
                artists = track.get("artists", [])
                artist_names = [a["name"].lower() for a in artists]
                if artist_name.lower() not in artist_names:
                    continue
                for a in artists:
                    if a["name"].lower() != artist_name.lower():
                        results.add(a["name"])
        except Exception as e:
            print(f"Erro ao processar faixas do álbum '{album.get('name', '')}': {e}")

    print(f"{artist_name}: {len(results)} colaboradores encontrados.")
    return results


def build_graph(first_artist, second_artist, start_year, end_year):
    print(f"Obtendo dados dos artistas...")
    id1, name1 = get_artist_info(first_artist)
    id2, name2 = get_artist_info(second_artist)

    print(f"Buscando colaborações de {name1}...")
    collabs1 = get_collaborations_by_period(id1, name1, start_year, end_year)
    print(f"Buscando colaborações de {name2}...")
    collabs2 = get_collaborations_by_period(id2, name2, start_year, end_year)

    country_cache = {}

    G = nx.Graph()
    all_collabs = collabs1.union(collabs2)

    for c in collabs1:
        G.add_edge(name1, c)
    for c in collabs2:
        G.add_edge(name2, c)

    print("Buscando colaborações de todos os colaboradores...")
    collab_data = {}
    total = len(all_collabs)

    for idx, artist in enumerate(all_collabs, 1):
        print(f"[{idx}/{total}] Processando {artist}...")
        try:
            artist_id, _ = get_artist_info(artist)
            collab_data[artist] = get_collaborations_by_period(artist_id, artist, start_year, end_year)
            time.sleep(0.3)
        except Exception as e:
            print(f"Erro ao buscar colabs de {artist}: {e}")
            collab_data[artist] = set()


    print("Verificando conexões diretas entre colaboradores...")
    for a1 in all_collabs:
        for a2 in all_collabs:
            if a1 != a2 and a2 in collab_data.get(a1, set()):
                G.add_edge(a1, a2)

    return G

def plot_graph(G, first_artist, second_artist, start_year, end_year):
    print("Plotando grafo...")
    plt.figure(figsize=(14, 9))
    pos = nx.spring_layout(G, seed=42, k=1.5)

    node_colors = []
    for node in G.nodes:
        if first_artist.lower() in node.lower():
            node_colors.append("orange")
        elif second_artist.lower() in node.lower():
            node_colors.append("red")
        else:
            node_colors.append("skyblue")

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000)
    nx.draw_networkx_edges(G, pos, edge_color="gray", width=1.5)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')

    plt.title(f"Colaborações: {first_artist} & {second_artist} ({start_year}-{end_year})", fontsize=16)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    nx.write_edgelist(G, f"grafo_colaboracoes_{first_artist}-{second_artist}-{start_year}-{end_year}.csv", delimiter=";")
    print(f"Grafo exportado para 'grafo_colaboracoes_{first_artist}-{second_artist}-{start_year}-{end_year}.csv'.")

def main():
    artist1 = input("Primeiro artista: ")
    artist2 = input("Segundo artista: ")
    start = int(input("Ano inicial: "))
    end = int(input("Ano final: "))

    G = build_graph(artist1, artist2, start, end)
    plot_graph(G, artist1, artist2, start, end)
    plot_interactive_graph(G, artist1, artist2, start, end)


if __name__ == "__main__":
    main()
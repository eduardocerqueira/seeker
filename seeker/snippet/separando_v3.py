#date: 2022-07-25T16:56:38Z
#url: https://api.github.com/gists/b042482bbdc944a098bf85c08404015f
#owner: https://api.github.com/users/fonseca-carlos

from sklearn.neighbors import NearestNeighbors

#gerando lista com as coordenadas xy de cada ponto
indexes=[]
for i in range(w):
    for j in range(h):
        indexes.append([i,j])

#para cada regiao, rodar o knn
for region in range(len(regions)):
    #inicializacao da classe
    knn=NearestNeighbors(n_neighbors=region_sizes[region+1])
    knn.fit(indexes)
    #encontrando os k pontos mais proximos
    points=knn.kneighbors([region_centers[region]],return_distance=False)
    #pegando as coordenadas de cada ponto, e transformando para terem o valor dessa regiao
    for point in points:
        line=point//w
        column=point%w
        area[line,column]=region+1
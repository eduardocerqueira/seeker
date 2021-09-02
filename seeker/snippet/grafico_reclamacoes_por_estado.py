#date: 2021-09-02T17:17:50Z
#url: https://api.github.com/gists/430e073356a2af111c6edd04e8a20140
#owner: https://api.github.com/users/RailanDeivid

# setando tamanho do gráfico
plt.subplots(figsize=(18, 8))
# atribuindo dados ao gráfico
sns.barplot(data=reclamacoes_por_estado,
            x='QUANTIDADE', 
            y='UF',
            color="red")
# setando titulo do gráfico
ax = plt.title("Número de reclamações por Estado" , fontsize=16)


# definindo variáveis de anotações do gráfico
n = range(reclamacoes_por_estado.UF.shape[0])
s = reclamacoes_por_estado.QUANTIDADE.values
# criando função para a visualização dos valores na barra
for i in range(len(s)):
    plt.annotate(s[i],
                 xy=(s[i]*1.01, n[i]), fontsize=11,
                 ha='left', 
                 va="center_baseline")
    
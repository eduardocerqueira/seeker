#date: 2022-03-03T16:51:48Z
#url: https://api.github.com/gists/1df3a806e1a3a600b389a148b17ea28e
#owner: https://api.github.com/users/MiguelOyarzo

#Se consideran todos los años de la base y sólo la categoría "Ambos Sexos" de la variable DIM1. Se ordenan los países según la tasa cruda de suicidio anual.
dftotal=df[df["Dim1"]=="Both sexes"].sort_values("FactValueNumeric", ascending=False)
dftotal.head(10)
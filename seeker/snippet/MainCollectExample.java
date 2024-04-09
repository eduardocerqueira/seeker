//date: 2024-04-09T17:04:25Z
//url: https://api.github.com/gists/e0ca2bf641b4e48cd44924ad1cd2c063
//owner: https://api.github.com/users/kevin-llps

void main() {
    List<Talk> result = talks.stream()
            .collect(Collector.of(
                    ArrayList::new,
                    (groups, element) -> {
                        //Si Talk "element" a un niveau de difficulté supérieur ou égal à celui du précédent Talk
                        //=> Ajouter "element" en tant que Talk valide (ArrayList)
                    },
                    (_, _) -> {
                        throw new UnsupportedOperationException("Cannot be parallelized");
                    }
            ));  
}
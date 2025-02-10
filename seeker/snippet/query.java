//date: 2025-02-10T16:48:52Z
//url: https://api.github.com/gists/aa30de16571f3024edb4a30dda028a13
//owner: https://api.github.com/users/bastide

    /**
     * Calcule le nombre d'articles commandés par un client
     * @param clientCode la clé du client
     */
    // Attention : SUM peut renvoyer NULL si on ne trouve pas d'enregistrement
    // On utilise COALESCE pour renvoyer 0 dans ce cas
    // http://www.h2database.com/html/functions.html#coalesce
    @Query("SELECT COALESCE(SUM(l.quantite), 0) FROM Ligne l WHERE l.commande.client.code = :clientCode")
    int nombreArticlesCommandesPar(String clientCode);

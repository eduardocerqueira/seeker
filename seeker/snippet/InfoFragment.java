//date: 2021-12-31T16:41:40Z
//url: https://api.github.com/gists/d0718539d6a43749775935492b2a112b
//owner: https://api.github.com/users/Garmich

    public void imprimirEtiqueta (String codi, Integer impresora) {
        Log.d("imprimirEtiqueta", "Codi: " + codi + " Impresora:" + impresora);

        Call<Response> call = articleInterface.imprimirEtiqueta(codi, impresora);

        call.enqueue(new Callback<Response>() {
            @Override public void onResponse (Call<Response> call, Response<Response> response) {
                Log.d("ON RESPONSE", "onResponse: ");
            }

            @Override public void onFailure (Call<Response> call, Throwable t) {
                Log.d("ON FAILURE", "onFailure: ");

                customToast(getContext(), "Error al imprimir etiqueta",
                            R.color.vermell, R.color.negre);
            }
        });
    }
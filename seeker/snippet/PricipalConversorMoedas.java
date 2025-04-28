//date: 2025-04-28T16:46:54Z
//url: https://api.github.com/gists/f9c76c5c4ee2e652c0556a729ae5f203
//owner: https://api.github.com/users/Gabslimah

public class PricipalConversorMoedas {
package conversor;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.HashMap;
import org.json.JSONObject;

    public class ConversorMoedas {
        private HashMap<String, Double> taxasCambio;

        public ConversorMoedas() {
            taxasCambio = new HashMap<>();
            obterTaxasDeCambio(); // Chama o método para obter as taxas de câmbio
        }

        private void obterTaxasDeCambio() {
            try {
                String url = "https://api.exchangerate-api.com/v4/latest/BRL"; // URL da API
                HttpURLConnection conn = (HttpURLConnection) new URL(url).openConnection();
                conn.setRequestMethod("GET");

                // Verifica se a requisição foi bem-sucedida
                if (conn.getResponseCode() == HttpURLConnection.HTTP_OK) {
                    BufferedReader in = new BufferedReader(new InputStreamReader(conn.getInputStream()));
                    String inputLine;
                    StringBuilder response = new StringBuilder();

                    while ((inputLine = in.readLine()) != null) {
                        response.append(inputLine);
                    }
                    in.close();

                    // Processar a resposta JSON
                    JSONObject jsonResponse = new JSONObject(response.toString());
                    JSONObject taxas = jsonResponse.getJSONObject("rates");

                    // Adiciona as taxas ao HashMap
                    for (String moeda : taxas.keySet()) {
                        taxasCambio.put("BRL-" + moeda, taxas.getDouble(moeda));
                        taxasCambio.put(moeda + "-BRL", 1 / taxas.getDouble(moeda)); // Taxa inversa
                    }

                    // Adicione as taxas para BTC e ETH se necessário
                    // Exemplo: taxasCambio.put("BTC-BRL", 200000.0); // Atualize conforme necessário

                } else {
                    System.out.println("Erro na requisição: " + conn.getResponseCode());
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        public double converter(double valor, String moedaOrigem, String moedaDestino) {
            String chave = moedaOrigem + "-" + moedaDestino; // Cria a chave para buscar a taxa
            Double taxa = taxasCambio.get(chave); // Obtém a taxa de câmbio

            if (taxa != null) {
                return valor * taxa; // Retorna o valor convertido
            }
            return 0; // Retorna 0 se a conversão não for suportada
        }
    }
}

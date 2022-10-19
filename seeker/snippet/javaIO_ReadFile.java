//date: 2022-10-19T17:19:04Z
//url: https://api.github.com/gists/5c63e5f359f5c2b304bc0c35d048cd35
//owner: https://api.github.com/users/Davi-Mendonca

public static void leituraPrimeiroExemplo() throws IOException {

    // Lendo o arquivo em bytes
    FileInputStream fis = new FileInputStream("lorem.txt");

    // Convertendo os bytes lidos do arquivo para caracteres
    InputStreamReader isr = new InputStreamReader(fis);

    BufferedReader br = new BufferedReader(isr);

    String linha = br.readLine();

    // Enquanto houverem linhas, elas serão impressas
    while (linha != null){
        System.out.println(linha);
        linha = br.readLine();
    }

    br.close();
}

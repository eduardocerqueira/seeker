//date: 2022-10-19T17:21:24Z
//url: https://api.github.com/gists/eda30aa052c108ab2041ce34be8e06fb
//owner: https://api.github.com/users/Davi-Mendonca

public void leituraSegundoExemplo() throws IOException {

    InputStream fis = new FileInputStream("lorem.txt");
    Reader isr = new InputStreamReader(fis);
    BufferedReader br = new BufferedReader(isr);

    String linha = br.readLine();

    while (linha != null) {
        System.out.println(linha);
        linha = br.readLine();
    }

    br.close();
}
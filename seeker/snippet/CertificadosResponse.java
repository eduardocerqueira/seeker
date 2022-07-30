//date: 2022-07-30T18:50:25Z
//url: https://api.github.com/gists/651a2e7d43f9043a90fa10a50221b016
//owner: https://api.github.com/users/danyllosoareszup

public class CertificadosResponse {

    private String nome;

    public CertificadosResponse(Certificado certificado) {
        this.nome = certificado.getNome();
    }

    @Deprecated
    public CertificadosResponse() {
    }

    public String getNome() {

        return nome;
    }
}
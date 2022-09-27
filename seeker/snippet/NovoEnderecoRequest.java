//date: 2022-09-27T17:05:21Z
//url: https://api.github.com/gists/8401fddd5a225f2f56c87dd38df825f9
//owner: https://api.github.com/users/danielmotazup

public class NovoEnderecoRequest implements DadosNovoEndereco {

    private String cep;

    private String logradouro;

    private String numero;

    private String complemento;

    public String getCep() {
        return cep;
    }

    public String getLogradouro() {
        return logradouro;
    }

    public String getNumero() {
        return numero;
    }

    public String getComplemento() {
        return complemento;
    }


    @Override
    public Endereco toModel() {
        return new Endereco(this.cep, this.logradouro, this.numero, complemento);
    }
}
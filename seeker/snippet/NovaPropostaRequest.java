//date: 2022-09-27T17:05:21Z
//url: https://api.github.com/gists/8401fddd5a225f2f56c87dd38df825f9
//owner: https://api.github.com/users/danielmotazup

public class NovaPropostaRequest implements DadosNovaProposta {

    private String nome;

    private String cpf;

    private NovoEnderecoRequest novoEnderecoRequest;

    public String getNome() {
        return nome;
    }

    public String getCpf() {
        return cpf;
    }

    public NovoEnderecoRequest getNovoEnderecoRequest() {
        return novoEnderecoRequest;
    }

    @Override
    public Proposta toModel() {
        return new Proposta(this.nome, this.cpf, this.novoEnderecoRequest.toModel());
    }
}
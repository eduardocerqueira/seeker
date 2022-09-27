//date: 2022-09-27T17:07:46Z
//url: https://api.github.com/gists/49d61839153911cbb82d4bb8853f0ab8
//owner: https://api.github.com/users/danielmotazup

@Service
public class CriaNovaProposta {

    @Autowired
    private SqlPropostaRepository sqlPropostaRepository;


    public Proposta cria(NovaPropostaRequest novaPropostaRequest) {

        var proposta = novaPropostaRequest.toModel();

        return sqlPropostaRepository.salva(proposta);

    }
}
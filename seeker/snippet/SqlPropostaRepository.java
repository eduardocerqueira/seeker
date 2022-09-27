//date: 2022-09-27T17:12:19Z
//url: https://api.github.com/gists/827470397581e03eafc921c5f657b2a7
//owner: https://api.github.com/users/danielmotazup

@Component
public class SqlPropostaRepository implements CadastraNovaPropostaRepository {

    @Autowired
    private PropostaRepository propostaRepository;

    @Override
    public Proposta salva(Proposta proposta) {

        propostaRepository.save(proposta);

        return proposta;

    }

   
}
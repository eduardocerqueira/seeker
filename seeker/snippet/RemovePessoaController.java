//date: 2022-07-26T16:57:48Z
//url: https://api.github.com/gists/8226d9a6b566c75844e6fceb9d317f94
//owner: https://api.github.com/users/joaobragazup

@RestController
public class RemovePessoaController {

    private final PessoaRepository pessoaRepository;

    public RemovePessoaController(PessoaRepository pessoaRepository) {
        this.pessoaRepository = pessoaRepository;
    }

    @DeleteMapping("/api/pessoa/{id}")
    @Transactional
    public ResponseEntity<?> remove(@PathVariable Long id){

        Pessoa pessoa = pessoaRepository.findById(id).orElseThrow(() -> {
            return new ResponseStatusException(HttpStatus.NOT_FOUND, "Pessoa nao cadastrada no sistema");
        });

        Integer calculaIdade = pessoa.calculaIdade();

        if (calculaIdade >= 18){
            throw new ResponseStatusException(HttpStatus.UNPROCESSABLE_ENTITY, "Impossivel deletar pessoa com mais de 18 anos");
        }

        pessoaRepository.delete(pessoa);


        return ResponseEntity.noContent().build();
    }
}

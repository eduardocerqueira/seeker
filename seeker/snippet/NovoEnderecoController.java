//date: 2022-07-20T17:06:50Z
//url: https://api.github.com/gists/74310bc0c0ab80224eae17e08265de59
//owner: https://api.github.com/users/henriquesousazup

@RestController
@RequestMapping("/zuppers/{idZupper}/enderecos")
public class NovoEnderecoController {

    private final EnderecoRepository enderecoRepository;
    private final ZupperRepository zupperRepository;

    public NovoEnderecoController(EnderecoRepository enderecoRepository, ZupperRepository zupperRepository) {
        this.enderecoRepository = enderecoRepository;
        this.zupperRepository = zupperRepository;
    }

    @PostMapping
    public ResponseEntity<Void> cadastrar(@PathVariable Long idZupper, @RequestBody @Valid EnderecoRequest request, UriComponentsBuilder uriComponentsBuilder) {

        Zupper zupper = zupperRepository.findById(idZupper).orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Zupper não encontrado."));

        Endereco novosEndereco = request.toModel(zupper);
        enderecoRepository.save(novosEndereco);

        URI location = uriComponentsBuilder.path("zuppers/{idZupper}/enderecos/{id}").buildAndExpand(zupper.getId(),novosEndereco.getId()).toUri();
        return ResponseEntity.created(location).build();
    }
}

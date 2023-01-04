//date: 2023-01-04T16:39:43Z
//url: https://api.github.com/gists/ec0613209620ea136903bc65697aca20
//owner: https://api.github.com/users/bjerat

@HttpExchange(
        url = "/characters",
        accept = MediaType.APPLICATION_JSON_VALUE,
        contentType = MediaType.APPLICATION_JSON_VALUE)
public interface CharacterClient {

    @GetExchange
    List<CharacterResponse> getByName(@RequestParam String lastName);

    @GetExchange("/{id}")
    Optional<CharacterResponse> getById(@PathVariable long id);

    @PutExchange
    CharacterResponse addCharacter(@RequestBody CharacterRequest request);

    @DeleteExchange("/{id}")
    void deleteById(@PathVariable long id);

}
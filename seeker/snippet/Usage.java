//date: 2023-01-04T16:52:12Z
//url: https://api.github.com/gists/7bbb4e714c38c5823f9a65a8041887b4
//owner: https://api.github.com/users/bjerat

CharacterResponse brandon = characterClient.addCharacter(new CharacterRequest("Brandon", "Stark"));

List<CharacterResponse> starks = characterClient.getByName("Stark");

Optional<CharacterResponse> eddardStark = characterClient.getById(1);

Optional<CharacterResponse> unknown = characterClient.getById(1337L); // empty

characterClient.deleteById(brandon.id());
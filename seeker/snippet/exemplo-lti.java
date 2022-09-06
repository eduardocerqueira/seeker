//date: 2022-09-06T17:12:36Z
//url: https://api.github.com/gists/c37a38413f5052742a6a1e8dae48cb89
//owner: https://api.github.com/users/bugorin

@Lti
@RequestMapping("/lti")
public String ltiEntry(
    LtiVerificationResult result, // Classe da lib, que retorna se é uma mensagem LTI valida, e se esta autenticada corretamente
    @RequestParam("oauth_consumer_key") String consumer_key, //parametro do LTI que identifica o client 
    @RequestParam("lis_person_contact_email_primary") Optional<String> email //parametro do LTI que identifica o email a ser logado
  ) {
  
    if(!result.getSuccess()) {
      //caso tenha uma falha de autenticação, lançamos uma exception
    }
  
  //logica de logar o usuario

}
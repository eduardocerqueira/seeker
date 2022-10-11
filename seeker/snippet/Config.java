//date: 2022-10-11T17:24:28Z
//url: https://api.github.com/gists/70f3997a64313df7158685263657ad69
//owner: https://api.github.com/users/akulinski

@Configuration
public class VaultConfig extends AbstractVaultConfiguration {

    @Value("${vault.token}")
    private String token;

    @Value("${vault.host}")
    private String host;

    @Value("${vault.port}")
    private int port;

    @Value("${vault.scheme}")
    private String scheme;

    @Override
    public VaultEndpoint vaultEndpoint() {
        final var vaultEndpoint = VaultEndpoint.create(host, port);
        vaultEndpoint.setScheme(scheme);
        return vaultEndpoint;
    }


    @Override
    public ClientAuthentication clientAuthentication() {
        return new TokenAuthentication(token);
    }
}
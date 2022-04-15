//date: 2022-04-15T16:54:59Z
//url: https://api.github.com/gists/0712ba671f6508d77dadca2178daf6ca
//owner: https://api.github.com/users/tirmizee


@Configuration
public class DatasourceConfig {

    @Bean
    @Primary
    public Datasource masterDatasource() {
        return new Datasource("master");
    }

    @Bean
    public Datasource slaveDatasource() {
        return new Datasource("slave");
    }

}



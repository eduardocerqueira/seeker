//date: 2022-04-18T16:58:10Z
//url: https://api.github.com/gists/a5df4ca5616f3f5eb9d5459a8495bd3e
//owner: https://api.github.com/users/afagundes

@Configuration
public class DatasourceConfiguration {

  @Bean
  @Primary
  @ConfigurationProperties(prefix="spring.datasource")
  public DataSource primaryDataSource() {
      return DataSourceBuilder.create().build();
  }

  @Bean
  @ConfigurationProperties(prefix="spring.secondDatasource")
  public DataSource secondaryDataSource() {
      return DataSourceBuilder.create().build();
  }
  
}
//date: 2024-10-25T14:40:25Z
//url: https://api.github.com/gists/cd3d1ddc9b908948807c126d8eac072e
//owner: https://api.github.com/users/BabujiR

@Configuration
@EnableCaching
public class CacheConfiguration {
    @Bean
    public JCacheManagerFactoryBean cacheManagerFactoryBean() throws Exception {
        JCacheManagerFactoryBean jCacheManagerFactoryBean = new JCacheManagerFactoryBean();
        jCacheManagerFactoryBean.setCacheManagerUri(new ClassPathResource("ehcache.xml").getURI());
        return jCacheManagerFactoryBean;
    }

    @Bean
    public CacheManager cacheManager() throws Exception {
        final JCacheCacheManager jCacheCacheManager = new JCacheCacheManager();
        jCacheCacheManager.setCacheManager(cacheManagerFactoryBean().getObject());
        return jCacheCacheManager;
    }
}
//date: 2023-06-06T17:06:55Z
//url: https://api.github.com/gists/58ffd20b1478ab7d5648c875db560261
//owner: https://api.github.com/users/ys2017rhein

package config;

import org.springframework.boot.autoconfigure.domain.EntityScan;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.Ordered;
import org.springframework.core.annotation.Order;
import org.springframework.dao.annotation.PersistenceExceptionTranslationPostProcessor;
import org.springframework.data.jpa.repository.config.EnableJpaRepositories;
import org.springframework.transaction.annotation.EnableTransactionManagement;

/**
 * JPA配置类
 */
@Order(Ordered.HIGHEST_PRECEDENCE)
@Configuration
//启用JPA事务管理
@EnableTransactionManagement(proxyTargetClass = true)
//启用JPA资源库并指定上面定义的接口资源库的位置
@EnableJpaRepositories(basePackages = "com.example.demo2.repository")
//指定定义实体的位置
@EntityScan(basePackages = "com.example.demo2.entity")
public class JpaConfiguration {
    @Bean
    PersistenceExceptionTranslationPostProcessor persistenceExceptionTranslationPostProcessor() {
        return new PersistenceExceptionTranslationPostProcessor();
    }

}

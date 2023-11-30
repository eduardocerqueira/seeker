//date: 2023-11-30T17:08:19Z
//url: https://api.github.com/gists/42cf15f7e82f395a729602c02f69b521
//owner: https://api.github.com/users/chaeeun-Han

package com.hamahama.pupmory.conf;

import com.querydsl.jpa.impl.JPAQueryFactory;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import javax.persistence.EntityManager;
import javax.persistence.PersistenceContext;

@Configuration
public class QuerydslConfig {

    @PersistenceContext
    private EntityManager entityManager;

    @Bean
    public JPAQueryFactory jpaQueryFactory() {
        return new JPAQueryFactory(entityManager);
    }
}

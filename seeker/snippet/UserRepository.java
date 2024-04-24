//date: 2024-04-24T17:09:06Z
//url: https://api.github.com/gists/2ed6b79193c8f7c58abaf9c4d815e68c
//owner: https://api.github.com/users/rog3r

package br.com.residencia18.api.repository;

import java.util.Optional;

import org.springframework.jdbc.core.simple.JdbcClient;
import org.springframework.jdbc.support.GeneratedKeyHolder;
import org.springframework.jdbc.support.KeyHolder;
import org.springframework.stereotype.Repository;
import org.springframework.transaction.annotation.Transactional;

import br.com.residencia18.api.entity.User;
import lombok.RequiredArgsConstructor;

@Repository
@RequiredArgsConstructor
public class UserRepository {

    private final JdbcClient jdbcClient;

    @Transactional
    public Long saveUser(User user) {
        var insertQuery = """
                INSERT INTO users(username, password, email, role) 
                VALUES(?, ?, ?, ?)
                """;
        KeyHolder keyHolder = new GeneratedKeyHolder();
        jdbcClient.sql(insertQuery)
                .param(1, user.getUsername())
                .param(2, user.getPassword())
                .param(3, user.getEmail())
                .param(4, user.getRole())
                .update();
        return keyHolder.getKeyAs(Long.class);
    }

    @Transactional(readOnly = true)
    public Optional<User> findByUsername(String username) {
        var findQuery = "SELECT id, username, password, role, email FROM users WHERE username=: "**********"
        return jdbcClient.sql(findQuery).param("username", username).query(User.class).optional();
    }
}

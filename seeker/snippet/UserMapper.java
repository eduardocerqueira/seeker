//date: 2024-04-24T17:08:01Z
//url: https://api.github.com/gists/695dc9f5aade7212a2419aef57148ea0
//owner: https://api.github.com/users/rog3r

package br.com.residencia18.api.mapper;

import org.springframework.stereotype.Service;

import br.com.residencia18.api.dto.RegisterRequest;
import br.com.residencia18.api.entity.User;

@Service
public class UserMapper {

    public User fromRegisterRequest(RegisterRequest registerRequest) {
        return User.builder()
                .email(registerRequest.email())
                .username(registerRequest.username())
                .password(registerRequest.password())
                .role("ROLE_USER")
                .build();
    }
}
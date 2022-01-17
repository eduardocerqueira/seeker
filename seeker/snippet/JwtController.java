//date: 2022-01-17T17:14:17Z
//url: https://api.github.com/gists/e049847d8959d36340b98e5e5c1384c3
//owner: https://api.github.com/users/SoberGrim

package com.example.front.jwt;

import com.example.front.controller.RestService;
import org.json.JSONArray;
import org.json.JSONObject;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.*;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.security.Principal;
import java.util.*;

import static com.example.front.jwt.CookieUtils.*;
import static com.example.front.jwt.JwtUtils.*;


@Controller
@RequestMapping("/api")
public class JwtController {


    @PostMapping("/auth")
    public void jwtAuthPost(HttpServletResponse response, HttpServletRequest request, @RequestBody String credentials)
    {
        //todo проверить корректность при неверном credentials
        String usernameAuth = StringUtils.parse(credentials, "username=","&");
        String passwordAuth = StringUtils.parse(credentials, "password=","\n");

        String jsonData = RestService.getJSON("http://localhost/api/authserver?username="+usernameAuth+"&password="+passwordAuth).getBody();

        //todo корректно ли преобразование
        final JSONObject user = new JSONObject(jsonData);

        Long id = Long.parseLong(user.getString("id"));
        String username = user.getString("username");
        Collection<GrantedAuthority> roles = getRolesFromJson(user.getJSONArray("roles"));

        //TODO !!! remove hardcoded roles
        String accessToken = generateAccessToken(id, username, roles, 300);
        RefreshToken refreshToken = generateRefreshToken(id, username, 60*60*24*30);
        response.addCookie(setCookie("JWT", accessToken,300));
        response.addCookie(setCookie("JWR", refreshToken.getToken(),60*60*24*30));

        AuthUser.authUser(request, username, roles);
    }

    //http://localhost/api/logout
    @GetMapping("/logout")
    public String jwtLogout(HttpServletResponse response, Principal pr)
    {
        String username = pr.getName();
        if (username!=null) {
            response.addCookie(setCookie("JWT", "",0));
            response.addCookie(setCookie("JWR", "",0));
            SecurityContextHolder.clearContext();
        }
        return "redirect:/login";
    }

}
//date: 2023-03-23T16:56:57Z
//url: https://api.github.com/gists/dc248ab1229eea828f8ae74d1eeef5de
//owner: https://api.github.com/users/Kolman-Freecss

package com.kolmanfreecss.domain.auth;

import io.jsonwebtoken.Claims;
import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.SignatureAlgorithm;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.stereotype.Component;

import java.io.Serializable;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;

@Component
public class JwtTokenUtil implements Serializable {
    
    @Value("${jwt.secret}")
    private String secret;
    
    @Value("${jwt.expiration}")
    private Long JWT_TOKEN_VALIDITY;

    public Date getExpirationDateFromToken(String token) {
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"r "**********"e "**********"t "**********"u "**********"r "**********"n "**********"  "**********"g "**********"e "**********"t "**********"C "**********"l "**********"a "**********"i "**********"m "**********"F "**********"r "**********"o "**********"m "**********"T "**********"o "**********"k "**********"e "**********"n "**********"( "**********"t "**********"o "**********"k "**********"e "**********"n "**********", "**********"  "**********"C "**********"l "**********"a "**********"i "**********"m "**********"s "**********": "**********": "**********"g "**********"e "**********"t "**********"E "**********"x "**********"p "**********"i "**********"r "**********"a "**********"t "**********"i "**********"o "**********"n "**********") "**********"; "**********"
    }

    public <T> T getClaimFromToken(String token, Function<Claims, T> claimsResolver) {
        final Claims claims = "**********"
        return claimsResolver.apply(claims);
    }
    
    private Claims getAllClaimsFromToken(String token) {
        return Jwts.parser().setSigningKey(secret).parseClaimsJws(token).getBody();
    }

    private Boolean isTokenExpired(String token) {
        final Date expiration = "**********"
        return expiration.before(new Date());
    }

    public String generateToken(UserDetails userDetails) {
        Map<String, Object> claims = new HashMap<>();
        return doGenerateToken(claims, userDetails.getUsername());
    }
    
    private String doGenerateToken(Map<String, Object> claims, String subject) {

        return Jwts.builder().setClaims(claims).setSubject(subject).setIssuedAt(new Date(System.currentTimeMillis()))
                .setExpiration(new Date(System.currentTimeMillis() + JWT_TOKEN_VALIDITY * 1000))
                .signWith(SignatureAlgorithm.HS512, secret).compact();
    }
    
}

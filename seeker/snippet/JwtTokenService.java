//date: 2022-10-11T17:18:59Z
//url: https://api.github.com/gists/333b9c5755434ae62b22aa0e13517be0
//owner: https://api.github.com/users/akulinski


@Service
public class JwtTokenService {

    @Value("${jwt.secret}")
    private String secret;


    public String generateTokenFromUserDetails(UserDetails userDetails) {
        Map<String, Object> claims = new HashMap<>();
        return generateTokenWithSubject(claims, userDetails.getUsername());
    }

    private String generateTokenWithSubject(Map<String, Object> claims, String subject) {
        return Jwts.builder().setClaims(claims).setSubject(subject).setIssuedAt(new Date(System.currentTimeMillis()))
                .setExpiration(new Date(System.currentTimeMillis() + JWT_TOKEN_VALIDITY * 1000))
                .signWith(SignatureAlgorithm.HS512, secret).compact();
    }
}
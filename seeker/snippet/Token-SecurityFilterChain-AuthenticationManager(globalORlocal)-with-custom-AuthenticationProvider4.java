//date: 2023-01-27T16:58:17Z
//url: https://api.github.com/gists/ca66d7983e254149f6439354ac42402c
//owner: https://api.github.com/users/zzpzaf



@Configuration
@EnableWebSecurity
public class CustomSecurityConfiguration {

    @Autowired
    private AuthenticationConfiguration authenticationConfiguration;


    @Bean
    public SecurityFilterChain filterChain1(HttpSecurity http) throws Exception {

        http
            .csrf().disable()
            .exceptionHandling()
            .authenticationEntryPoint((request, response, authException) -> {
                                        response.setHeader("WWW-Authenticate", "Basic realm=SignIn");
                                       })
            .and()
                .sessionManagement()
                .sessionCreationPolicy(SessionCreationPolicy.STATELESS)
            .and()

            .authorizeHttpRequests(authorize -> authorize.requestMatchers(HttpMethod.POST, "/auth/signup").permitAll()
                                                     .requestMatchers(HttpMethod.GET, "/auth/signin").authenticated())

            .authorizeHttpRequests(authorize -> authorize.requestMatchers("/users").hasRole("ADMIN")
                                                     .requestMatchers("/items").hasAnyRole("ADMIN", "USER")
            )
            ;
           

        // http.addFilter(new BasicAuthenticationFilter(authenticationConfiguration.getAuthenticationManager())); 
        http.addFilterBefore( new CustomRequestHeaderTokenFilter(authenticationConfiguration.getAuthenticationManager()), UsernamePasswordAuthenticationFilter.class);    // OK ---

        return http.build();
    } 


}
//date: 2025-04-22T16:58:33Z
//url: https://api.github.com/gists/4d4300041d82ab7d306ccba7d789bb29
//owner: https://api.github.com/users/ArcureDev

@Configuration
@EnableWebSecurity
public interface SecurityConfig {
  @Bean
  public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
      http.formLogin(form -> form.loginPage("/login"))
              .logout(logoutConfig ->
                      logoutConfig.logoutUrl("/logout")
              )
              .authenticationManager(authenticationManager(http))
              .csrf(csrf -> csrf
                  .csrfTokenRepository(CookieCsrfTokenRepository.withHttpOnlyFalse())
                  .csrfTokenRequestHandler(new CsrfTokenRequestAttributeHandler())
              )
              .addFilterAfter(new CsrfCookieFilter(), BasicAuthenticationFilter.class);
      return http.build();
  }

  private static final class CsrfCookieFilter extends OncePerRequestFilter {
      @Override
      protected void doFilterInternal(
          HttpServletRequest request, HttpServletResponse response, FilterChain filterChain
      ) throws ServletException, IOException {
          CsrfToken csrfToken = "**********"
          csrfToken.getToken();

          filterChain.doFilter(request, response);
      }
  }
}Chain.doFilter(request, response);
      }
  }
}
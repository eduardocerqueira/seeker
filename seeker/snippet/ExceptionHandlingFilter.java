//date: 2022-06-06T17:03:45Z
//url: https://api.github.com/gists/5f95aac423df69a5d2a440acd658edf3
//owner: https://api.github.com/users/artemptushkin

@Component
@Profile("servlet")
@RequiredArgsConstructor
@ConditionalOnWebApplication(type = ConditionalOnWebApplication.Type.SERVLET)
public class ExceptionHandlingFilter extends OncePerRequestFilter {
    private final ServletExceptionHandler exceptionHandler;
    /**
     * naming this differently than _objectMapper_ you give a chance your code to pass a specific object mapper by the qualifier
     * the field name will be considered as the name of the bean
     */
    private final ObjectMapper exceptionHandlerObjectMapper;

    @Override
    protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response, FilterChain filterChain) throws IOException {
        try {
            filterChain.doFilter(request, response);
        } catch (Exception e) {
            ResponseEntity<ErrorResponse> responseEntity = exceptionHandler.handleException(e);
            writeResponseEntity(responseEntity, response);
        }
    }

    private void writeResponseEntity(ResponseEntity<ErrorResponse> responseEntity, HttpServletResponse response) throws IOException {
        PrintWriter out = response.getWriter();
        ErrorResponse error = responseEntity.getBody();
        response.setContentType(MediaType.APPLICATION_JSON_VALUE);
        response.setStatus(responseEntity.getStatusCodeValue());
        out.print(exceptionHandlerObjectMapper.writeValueAsString(error));
        out.flush();
    }
}
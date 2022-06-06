//date: 2022-06-06T17:06:58Z
//url: https://api.github.com/gists/964e7621c9b3eb0b340364f8b8f335e0
//owner: https://api.github.com/users/artemptushkin

@Slf4j
@ControllerAdvice
@Profile("servlet")
@RequiredArgsConstructor
@ConditionalOnWebApplication(type = ConditionalOnWebApplication.Type.SERVLET)
public class ServletExceptionHandler {
    private final Map<Class<? extends Exception>, HttpStatus> exceptionToStatusCode;
    private final HttpStatus defaultStatus;

    @ExceptionHandler(CustomExceptionInFilter.class)
    public ResponseEntity<ErrorResponse> handleCorrelationIdMalformedException(CustomExceptionInFilter ex) {
        return this.handleException(ex);
    }

    @ExceptionHandler(Exception.class)
    public ResponseEntity<ErrorResponse> handleException(Exception ex) {
        HttpStatus status = exceptionToStatusCode.getOrDefault(ex.getClass(), defaultStatus);
        ErrorResponse errorResponse = ErrorResponse
                .builder()
                .message(ex.getMessage())
                .code(status.value())
                .build();
        log.error("Exception has been occurred", ex);
        return new ResponseEntity<>(errorResponse, status);
    }
}
//date: 2022-06-06T17:12:07Z
//url: https://api.github.com/gists/75109f1799fe54ccf504c45fe4d16de8
//owner: https://api.github.com/users/artemptushkin

@Bean
public Map<Class<? extends Exception>, HttpStatus> exceptionToStatusCode() {
    return Map.of(
            CustomExceptionInController.class, HttpStatus.BAD_REQUEST,
            CustomExceptionInFilter.class, HttpStatus.BAD_REQUEST
    );
}
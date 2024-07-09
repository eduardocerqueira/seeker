//date: 2024-07-09T17:08:59Z
//url: https://api.github.com/gists/12ab04d2ca4e2de5ba5433274f9885e6
//owner: https://api.github.com/users/enrinal

@Override
public <T> Mono<Boolean> createCache(HeaderCommonRequest headerCommonRequest, T value, String key,
                                     Duration timeout) {
    long startTime = System.nanoTime();
    return Mono.defer(() ->
            Mono.fromCallable(() -> Snappy.compress(JSONHelper.convertObjectToJsonInString(value)))
                    .flatMap(compressedData -> {
                        redisTemplateByte.opsForValue().set(key, compressedData, timeout);
                        return Mono.just(true);
                    })
                    .then(Mono.just(true))
    )
            .elapsed()
            .hasElement();
}

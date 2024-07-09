//date: 2024-07-09T17:03:26Z
//url: https://api.github.com/gists/81141dba7ab189d65ab1acd1d4f8a749
//owner: https://api.github.com/users/enrinal

public <T> Mono<T> getCache(HeaderCommonRequest headerCommonRequest, String key, Class<T> clazz) {
    return Mono.defer(() -> {
        long startTime = System.nanoTime();
        return Mono.defer(() ->
                Optional.ofNullable(redisTemplateByte.opsForValue().get(key))
                        .map(Mono::just)
                        .orElseGet(Mono::empty))
                .elapsed()
                .flatMap(tupleData ->
                        Mono.fromCallable(() -> Snappy.uncompressString(tupleData.getT2()))
                                .flatMap(uncompressedData ->
                                        Mono.defer(() -> {
                                            log.debug("[{}] process time get cache key = {}, process time={} ms",
                                                    headerCommonRequest.getMandatoryRequest().getRequestId(), key,
                                                    tupleData.getT1());
                                            return Mono.just(
                                                    Objects.requireNonNull(
                                                            JSONHelper.convertJsonInStringToObject(uncompressedData, clazz)));
                                        })
                                )
                                .onErrorResume(throwable ->
                                        Mono.defer(() -> {
                                            cacheMonitoringUtil.logMetric(isSendMetric, "getCache", startTime, FAILED);
                                            log.error("[{}] getCache() - failed get this key={}. with this error={}",
                                                    headerCommonRequest.getMandatoryRequest().getRequestId(), key,
                                                    throwable);
                                            return Mono.error(
                                                    new BusinessLogicException(
                                                            CommonResponseCode.PARSE_DATA_ERROR.getCode(),
                                                            CommonResponseCode.PARSE_DATA_ERROR.getMessage()));
                                        }))
                )
                .doOnSuccess(res ->
                        cacheMonitoringUtil.logMetric(isSendMetric, "getCache", startTime, SUCCEED));
    });
}

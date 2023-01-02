//date: 2023-01-02T16:38:59Z
//url: https://api.github.com/gists/1166502d9e327aa20f9ee98c9a7ddfee
//owner: https://api.github.com/users/bjerat

retrySpec = Retry.backoff(
    inventoryClientProperties.getRetry().getMaxAttempts(),
    inventoryClientProperties.getRetry().getBackoffDuration()
)
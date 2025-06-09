//date: 2025-06-09T17:10:47Z
//url: https://api.github.com/gists/9885d9cdeb2ebc9848fdd2d929b53fc1
//owner: https://api.github.com/users/gagan-here

package com.example.integration.model;

import java.util.Map;

/**
 * Standardized result from a provider.
 */
public class VerificationResult {
    public enum Status {
        SUCCESS,
        ERROR,
        NEEDS_MORE_DATA
    }

    private Status status;
    private Map<String, Object> data;
    private String errorMessage;
    private String provider;  // e.g., "providerA", "providerB"

    public VerificationResult(Status status, Map<String, Object> data, String errorMessage, String provider) {
        this.status = status;
        this.data = data;
        this.errorMessage = errorMessage;
        this.provider = provider;
    }

    public Status getStatus() {
        return status;
    }
    public Map<String, Object> getData() {
        return data;
    }
    public String getErrorMessage() {
        return errorMessage;
    }
    public String getProvider() {
        return provider;
    }

    public boolean isSuccess() {
        return Status.SUCCESS.equals(status);
    }
    public boolean needsMoreData() {
        return Status.NEEDS_MORE_DATA.equals(status);
    }
}

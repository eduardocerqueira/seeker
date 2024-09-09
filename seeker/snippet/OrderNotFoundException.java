//date: 2024-09-09T16:59:28Z
//url: https://api.github.com/gists/a8fdd24edabd3ab01e2b5d5fd93d94a7
//owner: https://api.github.com/users/nkchauhan003

package com.cb.orderservice.domain.exception;

public class OrderNotFoundException extends RuntimeException {
    public OrderNotFoundException(String message) {
        super(message);
    }
}

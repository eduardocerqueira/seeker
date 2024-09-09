//date: 2024-09-09T17:01:21Z
//url: https://api.github.com/gists/c762b2b7b45cec9d7b0a2787b3db6ac4
//owner: https://api.github.com/users/nkchauhan003

package com.cb.orderservice.domain.model;

public enum OrderStatus {
    PENDING,
    PROCESSING,
    SHIPPED,
    DELIVERED,
    CANCELED,
    RETURNED;

    // Optional: Add methods to provide more information or functionality

    public String getDescription() {
        switch (this) {
            case PENDING:
                return "Order is pending confirmation.";
            case PROCESSING:
                return "Order is being processed.";
            case SHIPPED:
                return "Order has been shipped.";
            case DELIVERED:
                return "Order has been delivered.";
            case CANCELED:
                return "Order has been canceled.";
            case RETURNED:
                return "Order has been returned.";
            default:
                return "Unknown order status.";
        }
    }
}

//date: 2024-09-09T16:57:12Z
//url: https://api.github.com/gists/0a6e9fbbb960aa4500a06fb1cba96b54
//owner: https://api.github.com/users/nkchauhan003

package com.cb.orderservice.application.ports.input;

import com.cb.orderservice.domain.model.Order;

import java.util.Optional;

public interface GetOrderUseCase {
    Optional<Order> getOrder(Long orderId);
}

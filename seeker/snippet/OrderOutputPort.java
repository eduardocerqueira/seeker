//date: 2024-09-09T16:58:30Z
//url: https://api.github.com/gists/d572a248c4f1a65c9c0d9a4eea63a147
//owner: https://api.github.com/users/nkchauhan003

package com.cb.orderservice.application.ports.output;

import com.cb.orderservice.domain.model.Order;

import java.util.Optional;

public interface OrderOutputPort {
    Order save(Order order);

    Optional<Order> findById(Long orderId);
}

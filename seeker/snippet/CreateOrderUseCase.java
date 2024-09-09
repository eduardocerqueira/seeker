//date: 2024-09-09T16:56:23Z
//url: https://api.github.com/gists/cccf5dc66caca11503b460a0d3b75de5
//owner: https://api.github.com/users/nkchauhan003

package com.cb.orderservice.application.ports.input;

import com.cb.orderservice.domain.model.Order;

public interface CreateOrderUseCase {
    Order createOrder(Order order);
}

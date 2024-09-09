//date: 2024-09-09T17:00:11Z
//url: https://api.github.com/gists/768079e52335410c3de4d855171e3b57
//owner: https://api.github.com/users/nkchauhan003

package com.cb.orderservice.domain.model;

import lombok.AllArgsConstructor;
import lombok.Data;

import java.util.List;

@Data
@AllArgsConstructor
public class Order {
    private Long id;
    private List<OrderItem> items;
    private OrderStatus status;
}

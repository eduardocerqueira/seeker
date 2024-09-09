//date: 2024-09-09T17:00:48Z
//url: https://api.github.com/gists/74d3feb9b02ea58a2dcda43829bb2835
//owner: https://api.github.com/users/nkchauhan003

package com.cb.orderservice.domain.model;

import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class OrderItem {
    private Long productId;
    private int quantity;
}

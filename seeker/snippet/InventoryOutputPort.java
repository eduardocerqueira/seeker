//date: 2024-09-09T16:57:44Z
//url: https://api.github.com/gists/60f59bacd31e2704a9fa5a4228564afe
//owner: https://api.github.com/users/nkchauhan003

package com.cb.orderservice.application.ports.output;

public interface InventoryOutputPort {
    boolean checkInventory(Long productId, int quantity);
}

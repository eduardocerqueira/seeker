//date: 2023-08-29T16:52:48Z
//url: https://api.github.com/gists/cad42d0ff4af113a6d7dbe390d55836e
//owner: https://api.github.com/users/Patryk241194

package com.kodilla.ecommercee.crud;

import com.kodilla.ecommercee.domain.Cart;
import com.kodilla.ecommercee.domain.User;
import com.kodilla.ecommercee.repository.CartRepository;
import com.kodilla.ecommercee.repository.ProductRepository;
import com.kodilla.ecommercee.repository.UserRepository;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import java.sql.Date;
import java.time.LocalDate;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertFalse;

@SpringBootTest
public class CartRepositoryTestSuite {

    @Autowired
    private CartRepository cartRepository;

    @Autowired
    private ProductRepository productRepository;

    @Autowired
    private UserRepository userRepository;

    @Test
    public void cartRepositoryCreateTestSuite() {
        // Given
        User user = new User();
        user.setEmail("user@gmail.com");
        user.setUsername("user");
        user.setPassword("password");
        user.setGeneratedKey("key");
        user.setExpirationDate(Date.valueOf(LocalDate.now().plusDays(1)));
        user.setBlocked(false);

        userRepository.save(user);

        Cart cart = new Cart();
        cart.setUser(user);
        cart.setCreated(LocalDate.now().minusDays(1));

        // When
        cartRepository.save(cart);
        List<Cart> carts = (List<Cart>) cartRepository.findAll();

        // Then
        assertFalse(carts.isEmpty());
    }

    @Test
    public void cartRepositoryCreateTestSuiteShort() {
        // Given
        Cart cart = new Cart();
        cart.setCreated(LocalDate.now().minusDays(1));

        // When
        cartRepository.save(cart);
        List<Cart> carts = (List<Cart>) cartRepository.findAll();

        // Then
        assertFalse(carts.isEmpty());
    }

}

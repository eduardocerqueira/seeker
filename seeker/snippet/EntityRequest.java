//date: 2025-06-09T17:06:25Z
//url: https://api.github.com/gists/f7dda6c139aad144182e9880af188a30
//owner: https://api.github.com/users/gagan-here

package com.example.integration.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * DTO representing a generic entity request.
 * This could be extended later for specific verification use cases.
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class EntityRequest {
    private String identifier;
    private String name;
    private String dob;
}

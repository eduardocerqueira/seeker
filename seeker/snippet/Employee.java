//date: 2024-08-16T16:37:12Z
//url: https://api.github.com/gists/84cffe6e32f08c7d08241dfeba3ec697
//owner: https://api.github.com/users/Raja696969

package com.example.employeemanagement.model;

import javax.persistence.*;
import javax.validation.constraints.Email;
import javax.validation.constraints.NotBlank;
import javax.validation.constraints.Size;

@Entity
public class Employee {
    
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @NotBlank(message = "Name is mandatory")
    @Size(min = 2, max = 100, message = "Name must be between 2 and 100 characters")
    private String name;

    @NotBlank(message = "Email is mandatory")
    @Email(message = "Email should be valid")
    private String email;

    @NotBlank(message = "Department is mandatory")
    private String department;

    // Getters and Setters
}

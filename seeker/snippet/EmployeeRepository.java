//date: 2024-08-16T16:37:12Z
//url: https://api.github.com/gists/84cffe6e32f08c7d08241dfeba3ec697
//owner: https://api.github.com/users/Raja696969

package com.example.employeemanagement.repository;

import com.example.employeemanagement.model.Employee;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface EmployeeRepository extends JpaRepository<Employee, Long> {
}

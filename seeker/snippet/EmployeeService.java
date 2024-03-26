//date: 2024-03-26T16:55:00Z
//url: https://api.github.com/gists/ef0efad253bc735516c908422bcab8b6
//owner: https://api.github.com/users/rog3r

package com.residencia18.springbootmanytomany.service;

import java.util.List;
import java.util.Optional;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import com.raven.springbootmanytomany.entity.Employee;
import com.raven.springbootmanytomany.repository.EmployeeRepository;

//@RequiredArgsConstructor
@Service
public class EmployeeService {

    @Autowired
    private EmployeeRepository employeeRepository;

    public List<Employee> findAll(){
        return employeeRepository.findAll();
    }

    public Employee create(Employee employee) {
        return employeeRepository.save(employee);
    }

    public Optional<Employee> findById(Long id) {
        return employeeRepository.findById(id);
    }

    public Optional<Employee> findByName(String name) {
        return employeeRepository.findByName(name);
    }  

    public Optional<Employee> update(Long id, Employee updatedEmployee) {
        return employeeRepository.findById(id)
            .map(employee -> {
                employee.setName(updatedEmployee.getName());
                employee.setEmail(updatedEmployee.getEmail());
                employee.setTechnicalSkill(updatedEmployee.getTechnicalSkill());
                return employeeRepository.save(employee);
            });
    }

    public void deleteEmployee(Long id) {
        employeeRepository.deleteById(id);
    }

}

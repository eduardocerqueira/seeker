//date: 2024-04-12T16:54:13Z
//url: https://api.github.com/gists/290cf621cfc00a38d128ba4758407f64
//owner: https://api.github.com/users/rog3r

package br.com.residencia18.api.web;

import java.util.List;

import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import br.com.residencia18.api.domain.Employee;
import br.com.residencia18.api.domain.EmployeeServiceV1;
import br.com.residencia18.api.domain.Project;
import br.com.residencia18.api.domain.ProjectService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;

@RequiredArgsConstructor
@RequestMapping("/v1/employees")
@RestController
public class EmployeeControllerV1 {
    
    @Qualifier("employeeServiceV1")
    private final EmployeeServiceV1 employeeService;
    private final ProjectService projectService;

    @GetMapping
    public ResponseEntity<List<Employee>> getAllEmployees() {
        List<Employee> employees = employeeService.findAll();
        return ResponseEntity.ok(employees);
    }

    @PostMapping
    public ResponseEntity<Employee> createEmployee(@RequestBody @Valid Employee employee) {
        Employee savedEmployee = employeeService.create(employee);
        return new ResponseEntity<>(savedEmployee, HttpStatus.CREATED);
    }

    @GetMapping("/{id}")
    public ResponseEntity<Employee> getEmployeeById(@PathVariable Long id) {
        return employeeService.findById(id)
                .map(ResponseEntity::ok)
                .orElseGet(() -> ResponseEntity.notFound().build());
    }

    @PutMapping("/{id}")
    public ResponseEntity<Employee> updateEmployee(@PathVariable Long id, @RequestBody @Valid Employee employee) {
        return employeeService.update(id, employee)
                .map(ResponseEntity::ok)
                .orElseGet(() -> ResponseEntity.notFound().build());
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<Void> deleteEmployee(@PathVariable Long id) {
        employeeService.deleteEmployee(id);
        return ResponseEntity.noContent().build();
    }

    @PostMapping("/{employeeId}/assign/{projectId}")
    public ResponseEntity<Void> assignEmployeeToProject(@PathVariable Long employeeId, @PathVariable Long projectId) {
        Project project = projectService.findById(projectId).orElseThrow(() -> new RuntimeException("Project not found"));
        Employee employee = employeeService.findById(employeeId).orElseThrow(() -> new RuntimeException("Employee not found"));
        
        project.getEmployees().add(employee);
        projectService.create(project);
        
        return ResponseEntity.ok().build();
    }
}
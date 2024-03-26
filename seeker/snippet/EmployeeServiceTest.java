//date: 2024-03-26T16:58:27Z
//url: https://api.github.com/gists/04e9976957f1e286262f37653c9e5ce4
//owner: https://api.github.com/users/rog3r

package com.residencia18.springbootmanytomany.service;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.BDDMockito.*;
import java.util.Locale;
import java.util.Optional;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import com.github.javafaker.Faker;
import com.raven.springbootmanytomany.entity.Employee;
import com.raven.springbootmanytomany.repository.EmployeeRepository;

@ExtendWith(MockitoExtension.class)
public class EmployeeServiceTest {

    @Mock
    private EmployeeRepository employeeRepository;

    @InjectMocks
    private EmployeeService employeeService;

    private Employee employee;
    private Faker faker;

    @BeforeEach
    void setUp() {
        faker = new Faker(new Locale("en-US"));
        // Gerando dados fictícios com o FAKER
        employee = new Employee();
        employee.setId(1L); // Garantindo um ID para os testes de update
        employee.setName(faker.name().fullName());
        employee.setEmail(faker.internet().emailAddress());
        employee.setTechnicalSkill(faker.job().position());
    }

    @Test
    void testCreateFakeEmployee() {
        // Configura o Mockito para retornar o mesmo funcionario quando o repositório salvar qualquer funcionario
        given(employeeRepository.save(any(Employee.class))).willReturn(employee);

        // Ação:
        Employee savedEmployee = employeeService.create(employee);

        // Assert
        // Verifica se o método save do repositório foi chamado
        verify(employeeRepository).save(any(Employee.class));
        
        // verifica as propriedades do funcionario retornado para assegurar que elas correspondem ao esperado
        assertNotNull(savedEmployee, "O funcionario salvo não deve ser nulo");
        assertEquals(employee.getName(), savedEmployee.getName(), "O nome do funcionario não corresponde ao esperado");
        assertEquals(employee.getEmail(), savedEmployee.getEmail(), "O email do funcionario não corresponde ao esperado");
        assertEquals(employee.getTechnicalSkill(), savedEmployee.getTechnicalSkill(), "A habilidade técnica do funcionario não corresponde ao esperado");
    }

    @Test
    void shouldUpdateEmployeeSuccessfully() {
        Employee updatedEmployee = new Employee();
        updatedEmployee.setName("Jane Doe");
        updatedEmployee.setEmail("jane.doe@example.com");
        updatedEmployee.setTechnicalSkill("Spring Boot");
    
        // Assert
        when(employeeRepository.findById(employee.getId())).thenReturn(Optional.of(employee));
        when(employeeRepository.save(any(Employee.class))).thenReturn(updatedEmployee);
    
        // Ação:
        Optional<Employee> result = employeeService.update(employee.getId(), updatedEmployee);
    
        // Assert
        assertTrue(result.isPresent(), "O funcionario atualizado deve estar presente");
        assertEquals(updatedEmployee.getName(), result.get().getName(), "O nome do funcionario atualizado não corresponde ao esperado");
        assertEquals(updatedEmployee.getEmail(), result.get().getEmail(), "O email do funcionario atualizado não corresponde ao esperado");
        assertEquals(updatedEmployee.getTechnicalSkill(), result.get().getTechnicalSkill(), "A habilidade técnica do funcionario atualizado não corresponde ao esperado");
    
        verify(employeeRepository).findById(employee.getId());
        verify(employeeRepository).save(any(Employee.class));
    }
}

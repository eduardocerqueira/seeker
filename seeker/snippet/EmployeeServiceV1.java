//date: 2024-04-12T16:50:38Z
//url: https://api.github.com/gists/57763ba9407e2a1f1303114a11ca78d3
//owner: https://api.github.com/users/rog3r

import java.util.List;
import java.util.Optional;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.context.annotation.Primary;
import org.springframework.stereotype.Service;

@Service
@Primary //para indicar qual implementação deve ser preferida quando o Spring procura injetar um bean
@Qualifier("v1")
public class EmployeeServiceV1 {

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
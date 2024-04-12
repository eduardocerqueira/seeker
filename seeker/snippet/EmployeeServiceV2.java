//date: 2024-04-12T16:52:41Z
//url: https://api.github.com/gists/f601e385eec937ab5642366bf38e4ae4
//owner: https://api.github.com/users/rog3r

package br.com.residencia18.api.domain; 

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Service;

@Service
@Qualifier("v2")
public class EmployeeServiceV2 extends EmployeeServiceV1 {
    
    @Autowired
    private EmployeeRepository employeeRepository;

    public void deleteAllEmployees() {
        employeeRepository.deleteAll();
    }

}
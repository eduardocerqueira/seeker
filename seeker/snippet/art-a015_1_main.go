//date: 2023-10-13T17:09:21Z
//url: https://api.github.com/gists/8af77fb75016ea0c63ba6de0fc457bfb
//owner: https://api.github.com/users/xyrk

package main

import (
	"fmt"
)

type ExternalServiceInterface interface { //introduce interface to make it possible to test
	connectToExternalServiceAndGetCommissions(employeeID int) float32
	connectToExternalServiceAndGetBaseSalary(employeeID int) float32
}

type ExternalService struct { //struct that implements the interface
}

// we emulate the external service here just to make the code runnable. In reality this method suppost to connect to external service and get the data
func (es *ExternalService) connectToExternalServiceAndGetCommissions(employeeID int) float32 {
	return 500
}

// we emulate the external service here just to make the code runnable. In reality this method suppost to connect to external service and get the data
func (es *ExternalService) connectToExternalServiceAndGetBaseSalary(employeeID int) float32 {
	return 1000
}

// calculateSalary calculates the salary for the employee with the given ID. We didn't discuss the implementation of this function, but it's not important for the example.
func calculateSalary(employeeID int) float32 {
	externalService := &ExternalService{}
	totalSalary := calculateCommissions(employeeID, externalService) + calculateBaseSalary(employeeID, externalService)
	return totalSalary
}

// calculateCommissions takes the employee ID and externalServiceFunctions as parameters.
// The externalServiceFunctions is a struct that implements the ExternalServiceInterface interface.
func calculateCommissions(employeeID int, externalServiceFunctions ExternalServiceInterface) float32 {
	commissions := externalServiceFunctions.connectToExternalServiceAndGetCommissions(employeeID)
	return commissions
}

//calculateBaseSalary similar to calculateCommissions function
func calculateBaseSalary(employeeID int, externalServiceFunctions ExternalServiceInterface) float32 {
	baseSalary := externalServiceFunctions.connectToExternalServiceAndGetBaseSalary(employeeID)
	return baseSalary
}

func main() {
	employeeID := 2042
	salary := calculateSalary(employeeID)
	fmt.Printf("Salary for %d is: $%0.2f\n", employeeID, salary)
}

//date: 2024-09-03T17:01:38Z
//url: https://api.github.com/gists/28083a29785c2f6a3ec5816a8320a739
//owner: https://api.github.com/users/jorgecuza92

package com.capitalone.engagementexperience.reststepdefinitions.directdeposit;

import com.capitalone.banxoa.cucumber.AbstractRESTStepDefinition;
import com.capitalone.engagementexperience.model.ScenarioData;
import org.springframework.beans.factory.annotation.Autowired;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.concurrent.TimeUnit;

public class DirectDepositStepDefinitions extends AbstractRESTStepDefinition {

    // Autowired to inject ScenarioData, which contains transaction data
    @Autowired
    private ScenarioData scenarioData;

    // Method to set the enrollment status of Direct Deposit based on input and transaction time
    @And("^the account IsDirectDepositEnrolled value is (string)$")
    public void setScenarioDataToSupportDirectDepositEnrolment(String isDirectDepositEnrolledString) throws Exception {
        // Parse the input string to determine if direct deposit is enrolled
        boolean isDirectDepositEnrolled = Boolean.parseBoolean(isDirectDepositEnrolledString);

        // Fetch the last transaction time; replace with your method to get the correct date string if necessary
        String lastTransactionTime = scenarioData.getLastTransactionTime(); // Ensure this fetches the correct timestamp

        // Convert the lastTransactionTime from String to Date object
        DateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss"); // Adjust format if necessary
        Date transactionDate = dateFormat.parse(lastTransactionTime);
        Date currentDate = new Date(); // Get the current date and time

        // Calculate the difference in days between the current date and the last transaction date
        long differenceInMillis = currentDate.getTime() - transactionDate.getTime();
        long daysDifference = TimeUnit.MILLISECONDS.toDays(differenceInMillis);

        // Logic to determine if the enrollment status needs to be updated
        if (isDirectDepositEnrolled && daysDifference > 15) {
            // Update status to unenrolled if last transaction is older than 15 days
            scenarioData.setClientCorrelationId("unenrolled");
            scenarioData.setCustomerReferenceId("Direct Deposit inactive for over 15 days");
            System.out.println("Updated: Direct Deposit is not currently enrolled due to inactivity.");
        } else {
            // Keep enrollment as active if within 15 days
            scenarioData.setClientCorrelationId("enrolled");
            scenarioData.setCustomerReferenceId("Direct Deposit active");
        }
    }

    // Example of another existing method for reference, no changes needed
    @And("^invalid (string) is provided in scenario$")
    public void setInvalidFieldInScenarioData(String fieldName) {
        if (!fieldName.equals("CustomerReferenceId")) {
            scenarioData.setClientCorrelationId("invalid");
        }
    }
}
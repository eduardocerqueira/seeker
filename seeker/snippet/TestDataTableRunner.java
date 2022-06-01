//date: 2022-06-01T17:01:56Z
//url: https://api.github.com/gists/284afd452fd0bc6926cc82f31f8a0eca
//owner: https://api.github.com/users/rahulrathore44

package io.testing.tables.datatables;

import io.cucumber.testng.AbstractTestNGCucumberTests;
import io.cucumber.testng.CucumberOptions;


@CucumberOptions(
		dryRun = false,
		monochrome = true,
		features = {"src/test/resources/io/testing/tables/datatables/DataTables.feature"},
		plugin = { "pretty"},
		glue = {"classpath:io.testing.tables.datatables"}
		)
public class TestDataTableRunner extends AbstractTestNGCucumberTests {

}

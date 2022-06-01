//date: 2022-06-01T17:01:56Z
//url: https://api.github.com/gists/284afd452fd0bc6926cc82f31f8a0eca
//owner: https://api.github.com/users/rahulrathore44

package io.testing.tables.datatables;

import java.util.Map;

import io.cucumber.java.DataTableType;
import io.cucumber.java.en.Given;

public class StepDefinitions {
	
	@Given("The excel file name and location is given as")
	public void the_excel_file_name_and_location_is_given_as(IDataReader dataTable) {
	    // Write code here that turns the phrase above into concrete actions
	    // For automatic transformation, change DataTable to one of
	    // E, List<E>, List<List<E>>, List<Map<K,V>>, Map<K,V> or
	    // Map<K, List<V>>. E,K,V must be a String, Integer, Float,
	    // Double, Byte, Short, Long, BigInteger or BigDecimal.
	    //
	    // For other transformations you can register a DataTableType.
	   System.out.println(dataTable.getAllRows());
	   
	   
	}

	// 1. create a another method
	// 2. Parameter to the method will be a map object
	// 3. IdataReader will be return type
	// 4. @DataTableType
	
	@DataTableType
	public IDataReader excelToDataTable(Map<String, String> entry) { // [Excel= <fileName>, Location=<FileLocation> ..]
		ExcelConfiguration config = new ExcelConfiguration.ExcelConfigurationBuilder()
				.setFileName(entry.get("Excel"))
				.setFileLocation(entry.get("Location"))
				.setSheetName(entry.get("Sheet"))
				.setIndex(Integer.valueOf(entry.getOrDefault("Index", "0")))
				.build();
		return new ExcelDataReader(config);
				
	}

}
